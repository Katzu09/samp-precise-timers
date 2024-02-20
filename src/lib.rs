#![warn(clippy::pedantic)]
use amx_arguments::VariadicAmxArguments;
use log::{info, error};
use priority_queue::PriorityQueue;
use samp::amx::{Amx, AmxIdent};
use samp::cell::AmxString;
use samp::error::{AmxError, AmxResult};
use samp::plugin::SampPlugin;
use slab::Slab;
use std::cmp::Reverse;
use std::collections::hash_map::RandomState;
use std::convert::TryFrom;
use std::time::{Duration, Instant};
use timer::{Timer, TimerStaus};
use std::cell::RefCell;

mod amx_arguments;
mod timer;

thread_local! {
    pub static EXECUTING_TIMER: RefCell<Option<usize>> = RefCell::new(None);
}

/// The plugin and its data: a list of scheduled timers
struct PreciseTimers {
    timers: Slab<Timer>,
    queue: PriorityQueue<usize, Reverse<std::time::Instant>, RandomState>,
}

impl PreciseTimers {
    /// This function is called from PAWN via the C foreign function interface.
    /// It returns the timer identifier or 0 in case of failure.
    /// ```
    /// native SetPreciseTimer(const callback_name[], const interval, const bool:repeat, const types_of_arguments[]="", {Float,_}:...);
    /// ```
    #[samp::native(raw, name = "SetPreciseTimer")]
    pub fn create(&mut self, amx: &Amx, mut args: samp::args::Args) -> AmxResult<i32> {
        // Get the basic, mandatory timer parameters
        let callback_name = args.next::<AmxString>().ok_or(AmxError::Params)?;
        let interval = args
            .next::<i32>()
            .and_then(|ms| u64::try_from(ms).ok())
            .ok_or(AmxError::Params)
            .map(Duration::from_millis)?;
        let repeat = args.next::<bool>().ok_or(AmxError::Params)?;
        let passed_arguments = VariadicAmxArguments::from_amx_args(args, 3)?;

        // Find the callback by name and save its index
        let amx_callback_index = amx.find_public(&callback_name.to_string())?;
        let next_trigger = Instant::now() + interval;

        let timer = Timer {
            next_trigger,
            interval: if repeat { Some(interval) } else { None },
            passed_arguments,
            amx_identifier: AmxIdent::from(amx.amx().as_ptr()),
            amx_callback_index,
            scheduled_for_removal: false,
        };

        // Add the timer to the list. This is safe for Slab::retain() even if SetPreciseTimer was called from a timer's callback.
        let key: usize = self.timers.insert(timer);
        
        self.queue.push(key, Reverse(next_trigger));
        // The timer's slot in Slab<> incresed by 1, so that 0 signifies an invalid timer in PAWN
        let timer_number = key
            .checked_add(1)
            .and_then(|number| i32::try_from(number).ok())
            .ok_or(AmxError::Bounds)?;
        Ok(timer_number)
    }

    /// This function is called from PAWN via the C foreign function interface.
    /// Returns 0 if the timer does not exist.
    ///  ```
    /// native DeletePreciseTimer(timer_number)
    /// ```
    #[allow(clippy::unnecessary_wraps)]
    #[samp::native(name = "DeletePreciseTimer")]
    pub fn delete(&mut self, _: &Amx, timer_number: usize) -> AmxResult<i32> {
        // Subtract 1 from the passed timer_number (where 0=invalid) to get the actual Slab<> slot
        if let Some(timer) = self.timers.get_mut(timer_number - 1) {
            // We defer the removal so that we don't mess up the process_tick()->retain() iterator.
            timer.scheduled_for_removal = true;
            Ok(1)
        } else {
            Ok(0)
        }
    }

    /// This function is called from PAWN via the C foreign function interface.
    /// Returns 0 if the timer does not exist, 1 if removed.
    ///  ```
    /// native ResetPreciseTimer(timer_number, const interval, const bool:repeat)
    /// ```
    #[samp::native(name = "ResetPreciseTimer")]
    pub fn reset(
        &mut self,
        _: &Amx,
        timer_number: usize,
        interval: i32,
        repeat: bool,
    ) -> AmxResult<i32> {
        let key = timer_number - 1;
        if let Some(timer) = self.timers.get_mut(key) {
            let interval = u64::try_from(interval)
                .map(Duration::from_millis)
                .or(Err(AmxError::Params))?;

            let next_trigger = Instant::now() + interval;
            timer.interval = if repeat { Some(interval) } else { None };
            self.queue.change_priority(&key, Reverse(next_trigger));
            Ok(1)
        } else {
            Ok(0)
        }
    }
}

impl SampPlugin for PreciseTimers {
    fn on_load(&mut self) {
        info!("net4game.com/samp-precise-timers by Brian Misiak loaded correctly.");
    }

    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn process_tick(&mut self) {
        // Rust's Instant is monotonic and nondecreasing, even during NTP time adjustment.
        let now = Instant::now();
        let timer_keys_to_execute = Vec::new();

        // 💀⚠ Because of FFI with C, Rust can't notice the simultaneous mutation of self.timers, but the iterator could get messed up in case of
        // Slab::retain() -> Timer::trigger() -> PAWN callback/ffi which calls DeletePreciseTimer() -> Slab::remove.
        // That's why the DeletePreciseTimer() schedules timers for deletion instead of doing it right away.
        // Slab::retain() is, however, okay with inserting new timers during its execution, even in case of reallocation when over capacity.
        while let Some((&key, next_trigger)) = self.queue.peek() && next_trigger < now {
            let timer = self.timers.get_mut(key).unwrap();
            let interval = timer.interval;
            // Execute the callback:
            timer_keys_to_execute.push(key);

            if let Some(interval) = interval {
                let next_trigger = now + interval;
                self.queue.change_priority(key, Reverse(next_trigger));
            } else {
                self.timers.remove(*key);
                // the problem is, that damn callback might have injected a new timer
                // to the very beginning of the queue. So we'd be popping the wrong one.
                self.queue.pop().unwrap();
            }
        }

        for key in timer_keys_to_execute {
            let timer = self.timers.get_mut(key).unwrap();
            if let Err(err) = timer.execute_pawn_callback() {
                error!("Error executing timer callback: {}", err);
            }
        }
        
        loop {
            {
                let Some((key, &next_trigger)) = self.queue.peek() else {
                    continue;
                };
                if next_trigger > now {
                    continue;
                }
            }
        }
    }

    fn on_amx_unload(&mut self, unloaded_amx: &Amx) {
        self.timers
            .retain(|_, timer| !timer.was_scheduled_by_amx(unloaded_amx));
    }
}

samp::initialize_plugin!(
    natives: [
        PreciseTimers::delete,
        PreciseTimers::create,
        PreciseTimers::reset,
    ],
    {
        samp::plugin::enable_process_tick();

        // get the default samp logger (uses samp logprintf).
        let samp_logger = samp::plugin::logger().level(log::LevelFilter::Info); // logging info, warn and error messages

        let _ = fern::Dispatch::new()
            .format(|callback, message, record| {
                callback.finish(format_args!("samp-precise-timers {}: {}", record.level().to_string().to_lowercase(), message));
            })
            .chain(samp_logger)
            .apply();

        PreciseTimers {
            timers: Slab::with_capacity(1000),
            queue: PriorityQueue::with_capacity_and_default_hasher(1000),
        }
    }
);
