//  Copyright (c) 2026 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/init.hpp>
#include <hpx/modules/async_local.hpp>
#include <hpx/modules/coroutines.hpp>
#include <hpx/modules/execution_base.hpp>
#include <hpx/modules/futures.hpp>
#include <hpx/modules/synchronization.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/modules/threading_base.hpp>
#include <hpx/thread.hpp>

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>

// Predicate flips to true from another thread shortly before the deadline,
// but via a plain shared flag (no notify()/resume()) - so the waiting
// hpx::thread stays suspended on its scheduler timer for the full duration
// and only rechecks the predicate once that timer fires. Verifies the
// post-timeout recheck returns signaled (not timeout) when the predicate
// happens to already be true by then.
void test_sleep_predicate_true_at_deadline()
{
    std::atomic<bool> flag{false};
    std::atomic<bool> waiter_started{false};

    hpx::thread setter([&]() {
        auto const wait_deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!waiter_started.load(std::memory_order_acquire))
        {
            if (std::chrono::steady_clock::now() > wait_deadline)
            {
                HPX_TEST(false && "timed out waiting for waiter_started");
                return;
            }
            hpx::this_thread::yield();
        }
        hpx::this_thread::sleep_for(std::chrono::milliseconds(50));
        flag.store(true);    // or the mutex+notify variant for the second test
    });

    constexpr auto sleep_duration = std::chrono::milliseconds(500);
    std::chrono::steady_clock::time_point now;

    hpx::threads::thread_restart_state state{};
    hpx::thread waiter([&]() {
        now = std::chrono::steady_clock::now();
        state = hpx::execution_base::this_thread::agent().sleep_for(
            sleep_duration, [&flag, &waiter_started]() {
                waiter_started.store(true, std::memory_order_release);
                return flag.load();
            });
    });

    waiter.join();
    setter.join();

    HPX_TEST(state == hpx::threads::thread_restart_state::signaled);

    // no early-wake claim here: the flag doesn't trigger a resume, so the
    // waiter stays suspended on its timer for (approximately) the full
    // deadline before it rechecks the predicate and returns signaled.
    HPX_TEST(std::chrono::steady_clock::now() >=
        now + sleep_duration - std::chrono::milliseconds(50));
}

// Verifies that hpx::condition_variable::wait_for (which drives
// execution_agent::sleep_until under the hood) wakes a genuine hpx::thread via
// a real notify()-triggered resume once the predicate becomes true, rather than
// only returning at the deadline.
void test_wait_for_notify_before_timeout()
{
    hpx::mutex mtx;
    hpx::condition_variable cv;
    bool flag = false;
    std::atomic<bool> waiter_started{false};

    constexpr auto sleep_duration = std::chrono::milliseconds(500);
    std::chrono::steady_clock::time_point now;

    hpx::thread setter([&]() {
        // Ensure the waiter has already entered wait_for and is holding
        // the lock/checked the predicate once before setting flag.
        auto const wait_deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!waiter_started.load(std::memory_order_acquire))
        {
            if (std::chrono::steady_clock::now() > wait_deadline)
            {
                HPX_TEST(false && "timed out waiting for waiter_started");
                return;
            }
            hpx::this_thread::yield();
        }
        hpx::this_thread::sleep_for(std::chrono::milliseconds(50));
        {
            std::lock_guard<hpx::mutex> l(mtx);
            flag = true;
        }
        cv.notify_one();
    });

    bool result = false;
    hpx::thread waiter([&]() {
        now = std::chrono::steady_clock::now();
        std::unique_lock<hpx::mutex> l(mtx);
        result = cv.wait_for(l, sleep_duration, [&]() {
            waiter_started.store(true, std::memory_order_release);
            return flag;
        });
    });

    waiter.join();
    setter.join();

    HPX_TEST(result);

    // must return well before the full deadline - proves notify() actually
    // resumed the suspended waiter instead of it timing out and only then
    // observing the predicate is true.
    HPX_TEST(std::chrono::steady_clock::now() < now + sleep_duration / 2);
}

void test_sleep_with_predicate_hpx_thread_immediate()
{
    constexpr auto sleep_duration = std::chrono::milliseconds(100);
    hpx::threads::thread_restart_state state{};
    std::chrono::steady_clock::time_point now;

    hpx::thread waiter([&]() {
        now = std::chrono::steady_clock::now();
        state = hpx::execution_base::this_thread::agent().sleep_for(
            sleep_duration, []() { return true; });
    });
    waiter.join();

    HPX_TEST(state == hpx::threads::thread_restart_state::signaled);
    HPX_TEST(std::chrono::steady_clock::now() < now + sleep_duration);
}

void test_sleep_with_predicate_hpx_thread_timeout()
{
    constexpr auto sleep_duration = std::chrono::milliseconds(100);
    hpx::threads::thread_restart_state state{};
    std::chrono::steady_clock::time_point now;

    hpx::thread waiter([&]() {
        now = std::chrono::steady_clock::now();
        state = hpx::execution_base::this_thread::agent().sleep_for(
            sleep_duration, []() { return false; });
    });
    waiter.join();

    HPX_TEST(state == hpx::threads::thread_restart_state::timeout);
    HPX_TEST(now + sleep_duration <= std::chrono::steady_clock::now());
}

void test_arbitrated_timed_wait()
{
    using hpx::execution_base::agent_wait_state;
    using hpx::threads::thread_restart_state;
    using notification = agent_wait_state::notification;

    auto const agent = hpx::execution_base::this_thread::agent();
    auto const expired = std::chrono::steady_clock::now();

    auto predicate_state = std::make_shared<agent_wait_state>();
    HPX_TEST(agent.ref().suspend_until(
                 expired, predicate_state, []() { return true; },
                 "predicate wins") == thread_restart_state::signaled);
    HPX_TEST(predicate_state->notify(thread_restart_state::signaled) ==
        notification::stale);

    auto timeout_state = std::make_shared<agent_wait_state>();
    HPX_TEST(agent.ref().suspend_until(
                 expired, timeout_state, []() { return false; },
                 "timeout wins") == thread_restart_state::timeout);
    HPX_TEST(timeout_state->notify(thread_restart_state::signaled) ==
        notification::stale);

    // Before parking, notification must not schedule a resume that could
    // arrive in a later wait. It must also release a mutex needed by the
    // predicate without waiting for the native agent to suspend.
    auto notified_state = std::make_shared<agent_wait_state>();
    std::mutex predicate_mutex;
    std::atomic<bool> notified{false};
    std::thread notifier([&]() {
        std::lock_guard<std::mutex> l(predicate_mutex);
        auto const result =
            notified_state->notify(thread_restart_state::signaled);
        notified.store(true);
        HPX_TEST(result == notification::claimed);
        if (result == notification::resume)
            agent.resume();
    });
    while (!notified.load())
        std::this_thread::yield();
    HPX_TEST(agent.ref().suspend_until(
                 expired, notified_state,
                 [&]() {
                     std::lock_guard<std::mutex> l(predicate_mutex);
                     return true;
                 },
                 "notification wins") == thread_restart_state::signaled);
    notifier.join();
}

void test_timed_wait_followed_by_future()
{
    for (int i = 0; i != 10000; ++i)
    {
        hpx::promise<void> first;
        auto ready = first.get_future();
        hpx::post([p = std::move(first)]() mutable { p.set_value(); });
        ready.wait_for(std::chrono::milliseconds(1));
        ready.get();

        hpx::promise<void> second;
        auto next = second.get_future();
        hpx::post([p = std::move(second)]() mutable {
            for (int j = 0; j != 10; ++j)
                hpx::this_thread::yield();
            p.set_value();
        });
        try
        {
            next.get();
        }
        catch (hpx::exception const& e)
        {
            HPX_TEST_MSG(false, e.what());
            return;
        }
    }
}

int hpx_main()
{
    test_arbitrated_timed_wait();
    std::thread native_waiter(test_arbitrated_timed_wait);
    native_waiter.join();
    test_timed_wait_followed_by_future();
    test_sleep_predicate_true_at_deadline();
    test_wait_for_notify_before_timeout();
    test_sleep_with_predicate_hpx_thread_immediate();
    test_sleep_with_predicate_hpx_thread_timeout();

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv), 0,
        "HPX main exited with non-zero status");
    return hpx::util::report_errors();
}
