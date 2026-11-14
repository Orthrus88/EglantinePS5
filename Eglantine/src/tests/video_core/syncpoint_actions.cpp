// SPDX-License-Identifier: GPL-3.0-or-later

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "video_core/host1x/syncpoint_manager.h"

using Tegra::Host1x::SyncpointManager;

TEST_CASE("Already reached syncpoints run actions immediately with a zero handle",
          "[video_core][syncpoint_actions]") {
    SyncpointManager manager;
    u32 guest_calls{};
    u32 host_calls{};
    for (u32 value = 0; value < 3; ++value) {
        const auto guest = manager.RegisterGuestAction(0, value, [&] { ++guest_calls; });
        const auto host = manager.RegisterHostAction(0, value, [&] { ++host_calls; });
        REQUIRE(guest == 0);
        REQUIRE(host == 0);
        REQUIRE(guest_calls == value + 1);
        REQUIRE(host_calls == value + 1);
        manager.DeregisterGuestAction(0, guest);
        manager.DeregisterHostAction(0, host);
        manager.IncrementGuest(0);
        manager.IncrementHost(0);
        REQUIRE(guest_calls == value + 1);
        REQUIRE(host_calls == value + 1);
    }
}

TEST_CASE("Cancelling pending syncpoint actions prevents their callbacks",
          "[video_core][syncpoint_actions]") {
    SyncpointManager manager;
    u32 calls{};
    const auto guest = manager.RegisterGuestAction(0, 1, [&] { ++calls; });
    const auto host = manager.RegisterHostAction(0, 1, [&] { ++calls; });
    REQUIRE(guest != 0);
    REQUIRE(host != 0);
    REQUIRE(guest != host);
    manager.DeregisterGuestAction(0, guest);
    manager.DeregisterHostAction(0, host);
    manager.DeregisterGuestAction(0, guest);
    manager.DeregisterHostAction(0, host);
    manager.IncrementGuest(0);
    manager.IncrementHost(0);
    REQUIRE(calls == 0);
    REQUIRE(manager.GetGuestSyncpointValue(0) == 1);
    REQUIRE(manager.GetHostSyncpointValue(0) == 1);
}

TEST_CASE("Cancelling completed syncpoint actions is harmless",
          "[video_core][syncpoint_actions]") {
    SyncpointManager manager;
    u32 guest_calls{};
    u32 host_calls{};
    const auto guest = manager.RegisterGuestAction(0, 1, [&] { ++guest_calls; });
    const auto host = manager.RegisterHostAction(0, 1, [&] { ++host_calls; });
    manager.IncrementGuest(0);
    manager.IncrementHost(0);
    REQUIRE(guest_calls == 1);
    REQUIRE(host_calls == 1);
    for (u32 repeat = 0; repeat < 3; ++repeat) {
        manager.DeregisterGuestAction(0, guest);
        manager.DeregisterHostAction(0, host);
        manager.IncrementGuest(0);
        manager.IncrementHost(0);
    }
    REQUIRE(guest_calls == 1);
    REQUIRE(host_calls == 1);
}

TEST_CASE("Stale syncpoint handles cannot cancel replacement actions",
          "[video_core][syncpoint_actions]") {
    SyncpointManager manager;
    u32 guest_calls{};
    u32 host_calls{};
    SyncpointManager::ActionHandle previous_guest{};
    SyncpointManager::ActionHandle previous_host{};
    for (u32 target = 1; target <= 256; ++target) {
        const auto guest = manager.RegisterGuestAction(0, target, [&] { ++guest_calls; });
        const auto host = manager.RegisterHostAction(0, target, [&] { ++host_calls; });
        REQUIRE(guest > previous_guest);
        REQUIRE(host > previous_host);
        REQUIRE(guest != host);
        manager.DeregisterGuestAction(0, previous_guest);
        manager.DeregisterHostAction(0, previous_host);
        manager.IncrementGuest(0);
        manager.IncrementHost(0);
        REQUIRE(guest_calls == target);
        REQUIRE(host_calls == target);
        previous_guest = guest;
        previous_host = host;
    }
}

TEST_CASE("Syncpoint action IDs stay distinct across counters and host guest banks",
          "[video_core][syncpoint_actions]") {
    SyncpointManager manager;
    const auto stale = manager.RegisterGuestAction(0, 1, [] {});
    manager.IncrementGuest(0);
    u32 guest_calls{};
    u32 host_calls{};
    const auto guest = manager.RegisterGuestAction(1, 1, [&] { ++guest_calls; });
    const auto host = manager.RegisterHostAction(1, 1, [&] { ++host_calls; });
    REQUIRE(stale != guest);
    REQUIRE(stale != host);
    REQUIRE(guest != host);
    manager.DeregisterGuestAction(1, stale);
    manager.DeregisterHostAction(1, stale);
    manager.DeregisterGuestAction(1, host);
    manager.DeregisterHostAction(1, guest);
    manager.DeregisterGuestAction(0, guest);
    manager.DeregisterHostAction(0, host);
    manager.IncrementGuest(1);
    manager.IncrementHost(1);
    REQUIRE(guest_calls == 1);
    REQUIRE(host_calls == 1);
}

TEST_CASE("Syncpoint actions run outside the guard, in value order",
          "[video_core][syncpoint_actions]") {
    SyncpointManager manager;
    std::vector<u32> order;
    bool nested = false;
    manager.RegisterGuestAction(0, 2, [&] {
        order.push_back(2);
        manager.RegisterGuestAction(1, 1, [&nested] { nested = true; });
    });
    manager.RegisterGuestAction(0, 1, [&order] { order.push_back(1); });
    manager.RegisterGuestAction(0, 3, [&order] { order.push_back(3); });
    manager.IncrementGuest(0);
    manager.IncrementGuest(0);
    REQUIRE(order == std::vector<u32>{1, 2});
    manager.IncrementGuest(1);
    REQUIRE(nested);
    manager.IncrementGuest(0);
    REQUIRE(order == std::vector<u32>{1, 2, 3});
}

TEST_CASE("A syncpoint action can cancel itself and a later due action",
          "[video_core][syncpoint_actions]") {
    SyncpointManager manager;
    bool second = false;
    SyncpointManager::ActionHandle first{};
    const auto later = manager.RegisterGuestAction(0, 1, [&second] { second = true; });
    first = manager.RegisterGuestAction(0, 1, [&] {
        manager.DeregisterGuestAction(0, first);
        manager.DeregisterGuestAction(0, later);
    });
    manager.IncrementGuest(0);
    REQUIRE_FALSE(second);
}

TEST_CASE("A blocked syncpoint action holds up neither other syncpoints nor unrelated cancels",
          "[video_core][syncpoint_actions]") {
    SyncpointManager manager;
    std::atomic<bool> entered{false};
    std::atomic<bool> release{false};
    bool second = false;
    bool other = false;
    const auto handle = manager.RegisterGuestAction(0, 1, [&second] { second = true; });
    manager.RegisterGuestAction(0, 1, [&] {
        entered = true;
        while (!release) {
            std::this_thread::yield();
        }
    });
    std::thread incrementer{[&] { manager.IncrementGuest(0); }};
    while (!entered) {
        std::this_thread::yield();
    }
    manager.RegisterGuestAction(1, 1, [&other] { other = true; });
    manager.IncrementGuest(1);
    manager.IncrementHost(0);
    REQUIRE(other);
    manager.DeregisterGuestAction(0, handle);
    release = true;
    incrementer.join();
    REQUIRE_FALSE(second);
}

TEST_CASE("Cancelling a running syncpoint action returns once it is done",
          "[video_core][syncpoint_actions]") {
    SyncpointManager manager;
    std::atomic<bool> entered{false};
    std::atomic<bool> release{false};
    std::atomic<bool> done{false};
    const auto handle = manager.RegisterGuestAction(0, 1, [&] {
        entered = true;
        while (!release) {
            std::this_thread::yield();
        }
        done = true;
    });
    std::thread incrementer{[&] { manager.IncrementGuest(0); }};
    while (!entered) {
        std::this_thread::yield();
    }
    std::thread releaser{[&] {
        std::this_thread::sleep_for(std::chrono::milliseconds{20});
        release = true;
    }};
    manager.DeregisterGuestAction(0, handle);
    REQUIRE(done);
    releaser.join();
    incrementer.join();
}
