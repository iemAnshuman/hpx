//  Copyright (c) 2019 Thomas Heller
//  Copyright (c) 2026 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/execution_base/context_base.hpp>
#include <hpx/modules/coroutines.hpp>
#include <hpx/modules/functional.hpp>
#include <hpx/modules/timing.hpp>

#include <atomic>
#include <cstddef>
#include <memory>
#include <string>

namespace hpx::execution_base {

    /// Arbitration shared by the notifier and the deadline of one timed wait.
    HPX_CXX_CORE_EXPORT struct agent_wait_state
    {
        enum class notification
        {
            stale,
            claimed,
            resume
        };

        /// Select the only completion. Resume only if parking has begun.
        notification notify(threads::thread_restart_state reason) noexcept
        {
            auto const desired =
                reason == threads::thread_restart_state::timeout ?
                state::timeout :
                reason == threads::thread_restart_state::abort ?
                state::abort :
                state::signaled;
            auto expected = state_.load();
            while (expected == state::preparing || expected == state::waiting)
            {
                if (state_.compare_exchange_weak(expected, desired))
                {
                    return expected == state::waiting ? notification::resume :
                                                        notification::claimed;
                }
            }
            return notification::stale;
        }

        /// Commit to parking unless a completion was already selected.
        bool prepare_suspend() noexcept
        {
            auto expected = state::preparing;
            return state_.compare_exchange_strong(expected, state::waiting);
        }

        /// Return the reason that claimed this wait.
        threads::thread_restart_state reason() const noexcept
        {
            switch (state_.load())
            {
            case state::signaled:
                return threads::thread_restart_state::signaled;
            case state::timeout:
                return threads::thread_restart_state::timeout;
            case state::abort:
                return threads::thread_restart_state::abort;
            default:
                return threads::thread_restart_state::unknown;
            }
        }

    private:
        enum class state
        {
            preparing,
            waiting,
            signaled,
            timeout,
            abort
        };
        std::atomic<state> state_{state::preparing};
    };

    HPX_CXX_CORE_EXPORT struct agent_base
    {
        virtual ~agent_base() = default;

        [[nodiscard]] virtual std::string description() const = 0;

        [[nodiscard]] virtual context_base const& context() const noexcept = 0;

        virtual void yield(char const* desc) = 0;
        virtual bool yield_k(std::size_t k, char const* desc) = 0;
        virtual void suspend(char const* desc) = 0;
        /// Arbitrate predicate completion, notification, and the deadline.
        /// Once prepare_suspend succeeds, consume the selected wake by parking.
        virtual threads::thread_restart_state suspend_until(
            hpx::chrono::steady_time_point const& deadline,
            std::shared_ptr<agent_wait_state> const& state,
            hpx::move_only_function<bool()>&& wait_cond, char const* desc) = 0;
        virtual void resume(
            hpx::threads::thread_priority priority, char const* desc) = 0;
        virtual void abort(char const* desc) = 0;
        virtual threads::thread_restart_state sleep_for(
            hpx::chrono::steady_duration const& sleep_duration,
            hpx::move_only_function<bool()>&& wait_cond, char const* desc) = 0;
        virtual threads::thread_restart_state sleep_until(
            hpx::chrono::steady_time_point const& sleep_time,
            hpx::move_only_function<bool()>&& wait_cond, char const* desc) = 0;
    };
}    // namespace hpx::execution_base
