// SPDX-FileCopyrightText: Copyright 2025 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>

#include "common/logging.h"
#include "video_core/host1x/syncpoint_manager.h"

namespace Tegra {

namespace Host1x {

SyncpointManager::ActionHandle SyncpointManager::RegisterAction(
    std::atomic<u32>& syncpoint, std::list<RegisteredAction>& action_storage, u32 expected_value,
    std::function<void()>&& action) {
    if (syncpoint.load(std::memory_order_acquire) >= expected_value) {
        action();
        return {};
    }

    {
        std::scoped_lock lk(guard);
        if (syncpoint.load(std::memory_order_relaxed) < expected_value) {
            auto it = action_storage.begin();
            while (it != action_storage.end()) {
                if (it->expected_value >= expected_value) {
                    break;
                }
                ++it;
            }
            return action_storage.emplace(it, expected_value, ++last_action_id, std::move(action))
                ->id;
        }
    }
    // Reached while this waited for the guard: run it without the guard, as Increment does.
    action();
    return {};
}

void SyncpointManager::DeregisterAction(std::list<RegisteredAction>& action_storage,
                                        const ActionHandle& handle) {
    if (handle == 0) {
        return;
    }
    const auto matches = [&handle](const RegisteredAction& action) { return action.id == handle; };
    std::unique_lock lk(guard);
    if (const auto it = std::ranges::find_if(action_storage, matches); it != action_storage.end()) {
        action_storage.erase(it);
        return;
    }
    // An Increment may have taken it out as due. If it has not started it is dropped; if it is
    // running, the cancel returns once it is done. Its own thread cannot wait for itself.
    while (true) {
        const DueActions* running_in{};
        for (DueActions* due : due_actions) {
            if (const auto it = std::ranges::find_if(due->pending, matches);
                it != due->pending.end()) {
                due->pending.erase(it);
                return;
            }
            if (due->running == handle) {
                running_in = due;
            }
        }
        if (running_in == nullptr || running_in->thread == std::this_thread::get_id()) {
            return;
        }
        action_done_cv.wait(lk);
    }
}

void SyncpointManager::DeregisterGuestAction(u32 syncpoint_id, const ActionHandle& handle) {
    DeregisterAction(guest_action_storage[syncpoint_id], handle);
}

void SyncpointManager::DeregisterHostAction(u32 syncpoint_id, const ActionHandle& handle) {
    DeregisterAction(host_action_storage[syncpoint_id], handle);
}

void SyncpointManager::IncrementGuest(u32 syncpoint_id) {
    if (syncpoint_id >= NUM_MAX_SYNCPOINTS) {
        LOG_ERROR(HW_GPU, "Ignoring increment of invalid syncpoint {:#x}", syncpoint_id);
        return;
    }
    Increment(syncpoints_guest[syncpoint_id], wait_guest_cv, guest_action_storage[syncpoint_id]);
}

void SyncpointManager::IncrementHost(u32 syncpoint_id) {
    if (syncpoint_id >= NUM_MAX_SYNCPOINTS) {
        return;
    }
    Increment(syncpoints_host[syncpoint_id], wait_host_cv, host_action_storage[syncpoint_id]);
}

void SyncpointManager::WaitGuest(u32 syncpoint_id, u32 expected_value) {
    Wait(syncpoints_guest[syncpoint_id], wait_guest_cv, expected_value);
}

void SyncpointManager::WaitHost(u32 syncpoint_id, u32 expected_value) {
    Wait(syncpoints_host[syncpoint_id], wait_host_cv, expected_value);
}

void SyncpointManager::Increment(std::atomic<u32>& syncpoint, std::condition_variable& wait_cv,
                                 std::list<RegisteredAction>& action_storage) {
    std::unique_lock lk(guard);
    const u32 new_value{syncpoint.fetch_add(1, std::memory_order_acq_rel) + 1};
    wait_cv.notify_all();
    auto end = action_storage.begin();
    while (end != action_storage.end() && end->expected_value <= new_value) {
        ++end;
    }
    if (end == action_storage.begin()) {
        return;
    }
    // A composite action can block in the display backend. Run callbacks without the global
    // syncpoint guard so unrelated fences, registrations and cancellations continue to progress.
    DueActions due;
    due.pending.splice(due.pending.end(), action_storage, action_storage.begin(), end);
    due_actions.push_back(&due);
    while (!due.pending.empty()) {
        std::function<void()> action{std::move(due.pending.front().action)};
        due.running = due.pending.front().id;
        due.pending.pop_front();
        lk.unlock();
        action();
        action = nullptr;
        lk.lock();
        due.running = 0;
        action_done_cv.notify_all();
    }
    std::erase(due_actions, &due);
}

void SyncpointManager::Wait(std::atomic<u32>& syncpoint, std::condition_variable& wait_cv,
                            u32 expected_value) {
    const auto pred = [&]() { return syncpoint.load(std::memory_order_acquire) >= expected_value; };
    if (pred()) {
        return;
    }

    std::unique_lock lk(guard);
    wait_cv.wait(lk, pred);
}

} // namespace Host1x

} // namespace Tegra
