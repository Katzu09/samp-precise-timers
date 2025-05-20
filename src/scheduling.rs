use std::{cell::RefCell, time::Instant};

use fnv::FnvHashSet;
use slab::Slab;
use snafu::{ensure, OptionExt, Snafu};

use crate::{
    schedule::{Repeat, Schedule},
    timer::Timer,
};

struct State {
    /// A slotmap of timers. Stable keys.
    timers: Slab<Timer>,
    /// Always sorted queue of timers. Easy O(1) peeking and popping of the next scheduled timer.
    queue: Vec<Schedule>,
}

thread_local! {
    static STATE: RefCell<State> = RefCell::new(State {
        timers: Slab::with_capacity(1000),
        queue: Vec::with_capacity(1000),
    })
}

#[derive(Debug, Snafu)]
#[snafu(context(suffix(false)))]
pub(crate) enum TriggeringError {
    #[snafu(display("Unable to find timer in priority queue"))]
    TimerNotInQueue,
}

pub(crate) fn insert_and_schedule_timer(
    timer: Timer,
    get_schedule_based_on_key: impl FnOnce(usize) -> Schedule,
) -> usize {
    STATE.with_borrow_mut(|State { timers, queue }| {
        let key = timers.insert(timer);
        let schedule = get_schedule_based_on_key(key);
        let new_position = queue.partition_point(|s| s < &schedule);
        queue.insert(new_position, schedule);
        key
    })
}

pub(crate) fn delete_timer(timer_key: usize) -> Result<(), TriggeringError> {
    STATE.with_borrow_mut(|State { timers, queue }| {
        ensure!(timers.contains(timer_key), TimerNotInQueue);
        timers.remove(timer_key);
        queue.retain(|s| s.key != timer_key);
        Ok(())
    })
}

pub(crate) fn reschedule_timer(key: usize, new_schedule: Schedule) -> Result<(), TriggeringError> {
    STATE.with_borrow_mut(|State { queue, .. }| {
        let current_index = queue
            .iter()
            .position(|s| s.key == key)
            .context(TimerNotInQueue)?;
        let new_index = queue.partition_point(|s| s < &new_schedule);
        queue[current_index].next_trigger = new_schedule.next_trigger;
        queue[current_index].repeat = new_schedule.repeat;
        if new_index < current_index {
            queue[new_index..=current_index].rotate_right(1);
        } else if new_index > current_index {
            queue[current_index..=new_index].rotate_left(1);
        }
        Ok(())
    })
}

pub(crate) fn remove_timers(predicate: impl Fn(&Timer) -> bool) {
    STATE.with_borrow_mut(|State { timers, queue }| {
        let mut removed_keys = FnvHashSet::default();
        queue.retain(|&Schedule { key, .. }| {
            if predicate(&timers[key]) {
                removed_keys.insert(key);
                false
            } else {
                true
            }
        });
        for key in removed_keys {
            timers.remove(key);
        }
    });
}

/// 1. Reschedules (or deschedules) the timer
/// 2. While holding the timer, gives it to the closure
///    (which uses its data to push onto the amx stack)
/// 3. Frees state.
/// 4. Returns the result of the closure.
/// `timer_manipulator` must not borrow state
#[inline]
pub(crate) fn reschedule_next_due_and_then<T>(
    now: Instant,
    timer_manipulator: impl FnOnce(&Timer) -> T,
) -> Option<T> {
    STATE.with_borrow_mut(|State { timers, queue }| {
        let Some(scheduled @ &Schedule { key, .. }) = queue.last() else {
            return None;
        };
        if scheduled.next_trigger > now {
            return None;
        }

        if let Repeat::Every(interval) = scheduled.repeat {
            let next_trigger = now + interval;
            let old_position = queue.len() - 1;
            let new_position = queue.partition_point(|s| s.next_trigger >= next_trigger);
            queue[old_position].next_trigger = next_trigger;
            if new_position < old_position {
                queue[new_position..].rotate_right(1);
            } else {
                debug_assert_eq!(new_position, old_position);
            }

            let timer = timers.get_mut(key).expect("due timer should be in slab");
            Some(timer_manipulator(timer))
        } else {
            let descheduled = queue.pop().expect("due timer should be in queue");
            assert_eq!(descheduled.key, key);

            let timer = timers.remove(key);
            Some(timer_manipulator(&timer))
        }
    })
}

#[cfg(test)]
mod test {
    use std::{ptr::null_mut, time::{Duration, Instant}}; 

    use crate::{
        amx_arguments::{PassedArgument, VariadicAmxArguments}, 
        schedule::Repeat::{DontRepeat, Every},
        scheduling::{delete_timer, remove_timers, reschedule_timer, STATE, reschedule_next_due_and_then}, // Removed State import
        timer::Timer,
    };
    use super::{insert_and_schedule_timer, Schedule, TriggeringError}; // Added TriggeringError import

    fn reset_state() {
        STATE.with_borrow_mut(|state| {
            state.timers.clear();
            state.queue.clear();
        });
    }

    // Creates a timer with a specific ID stored in its arguments for testing.
    fn timer_with_id(id: i32) -> Timer {
        Timer {
            passed_arguments: VariadicAmxArguments {
                // Accessing inner field directly. Ensure VariadicAmxArguments is used as per its definition.
                // If VariadicAmxArguments::new(Vec<PassedArgument>) is better, adjust.
                // For now, assuming direct field access is okay for test setup based on its structure.
                inner: vec![PassedArgument::PrimitiveCell(id)],
            },
            amx_callback_index: samp::consts::AmxExecIdx::Continue,
            amx: samp::amx::Amx::new(null_mut(), 0),
        }
    }
    
    // Legacy empty_timer, now uses timer_with_id(0) or a distinct marker if needed.
    fn empty_timer() -> Timer {
        timer_with_id(-1) // Use a default ID for generic empty timers
    }

    fn noop(_timer: &Timer) {} // Simple closure for reschedule_next_due_and_then

    // Helper to create a Schedule for a timer that repeats every `interval_ms`
    // and its first trigger is `offset_ms` from now.
    fn repeating_schedule(key: usize, offset_ms: u64, interval_ms: u64) -> Schedule {
        Schedule {
            key,
            next_trigger: Instant::now() + Duration::from_millis(offset_ms),
            repeat: Every(Duration::from_millis(interval_ms)),
        }
    }

    // Helper to create a Schedule for a non-repeating timer
    // with its trigger `offset_ms` from now.
    fn non_repeating_schedule(key: usize, offset_ms: u64) -> Schedule {
        Schedule {
            key,
            next_trigger: Instant::now() + Duration::from_millis(offset_ms),
            repeat: DontRepeat,
        }
    }

    // Updated to take &State to avoid nested STATE.with_borrow_mut
    fn timer_keys_from_queue(queue: &Vec<Schedule>) -> Vec<usize> {
        queue.iter().map(|s| s.key).collect()
    }
    
    fn get_timer_id(timer: &Timer) -> Option<i32> {
        // Use the new public accessor method from VariadicAmxArguments
        timer.passed_arguments.get_first_primitive()
    }

    #[test]
    fn hello_original_logic_check() {
        reset_state();
        let first_key = insert_and_schedule_timer(timer_with_id(1), |k| repeating_schedule(k, 100, 1000)); 
        let second_key = insert_and_schedule_timer(timer_with_id(2), |k| repeating_schedule(k, 200, 1000));
        let third_key = insert_and_schedule_timer(timer_with_id(3), |k| repeating_schedule(k, 300, 1000)); 
        let fourth_key = insert_and_schedule_timer(timer_with_id(4), |k| non_repeating_schedule(k, 400));
        
        STATE.with_borrow(|state_ref| { // Use state_ref to avoid conflict
            assert_eq!(timer_keys_from_queue(&state_ref.queue), vec![fourth_key, third_key, second_key, first_key]);
        });

        let time_after_first_due = Instant::now() + Duration::from_millis(150);
        assert!(reschedule_next_due_and_then(time_after_first_due, noop).is_some());

        STATE.with_borrow(|state_ref| {
            let current_keys = timer_keys_from_queue(&state_ref.queue);
            assert_ne!(current_keys.last(), Some(&first_key));
            assert_eq!(state_ref.queue.len(), 4);
            assert!(state_ref.timers.contains(first_key));
        });
        
        let time_after_all_due = Instant::now() + Duration::from_millis(500);
        
        // Simplified processing loop for remaining timers
        // This loop will process timers as long as they are due at time_after_all_due
        while reschedule_next_due_and_then(time_after_all_due, noop).is_some() {
            // Loop until no more timers are immediately due at this 'now'
        }

        STATE.with_borrow(|state_ref| {
            assert!(!state_ref.timers.contains(fourth_key), "Non-repeating timer 'fourth_key' should be removed");
            assert!(state_ref.timers.contains(first_key)); // Rescheduled
            // second_key and third_key were repeating, should have been processed and rescheduled
            assert!(state_ref.timers.contains(second_key));
            assert!(state_ref.timers.contains(third_key));
            assert_eq!(state_ref.timers.len(), 3); 
        });
    }

    #[test]
    fn test_insert_and_schedule_timer_ordering() {
        reset_state();
        let now = Instant::now();

        // Insert timers with different next_trigger times
        let key1 = insert_and_schedule_timer(timer_with_id(1), |_k| Schedule { key: _k, next_trigger: now + Duration::from_millis(200), repeat: DontRepeat });
        let key2 = insert_and_schedule_timer(timer_with_id(2), |_k| Schedule { key: _k, next_trigger: now + Duration::from_millis(100), repeat: DontRepeat });
        let key3 = insert_and_schedule_timer(timer_with_id(3), |_k| Schedule { key: _k, next_trigger: now + Duration::from_millis(300), repeat: DontRepeat });

        STATE.with_borrow(|state| {
            assert!(state.timers.contains(key1));
            assert!(state.timers.contains(key2));
            assert!(state.timers.contains(key3));
            // Queue stores soonest timer at the end (for pop)
            // Order of keys in queue: key3 (300ms), key1 (200ms), key2 (100ms)
            let keys = timer_keys_from_queue(&state.queue);
            assert_eq!(keys, vec![key3, key1, key2]);
        });
    }

    #[test]
    fn test_insert_and_schedule_timer_identical_next_trigger() {
        reset_state();
        let now = Instant::now();
        let trigger_time = now + Duration::from_millis(100);

        let key1 = insert_and_schedule_timer(timer_with_id(1), |_k| Schedule { key: _k, next_trigger: trigger_time, repeat: DontRepeat });
        let key2 = insert_and_schedule_timer(timer_with_id(2), |_k| Schedule { key: _k, next_trigger: trigger_time, repeat: DontRepeat });

        STATE.with_borrow(|state| {
            assert!(state.timers.contains(key1));
            assert!(state.timers.contains(key2));
            // Order for identical next_trigger is stable based on insertion order relative to partition_point
            // If partition_point returns index where new element should be inserted to maintain order,
            // and it's stable, then inserting key2 after key1 with same time might place it before or after key1.
            // The important part is they are processed based on this order.
            // Vec::insert shifts elements, so if key1 is at index `i`, and key2 is inserted at `i` (or `i+1`),
            // their relative order due to `partition_point` behavior with equal elements matters.
            // `partition_point` finds first element `s` for which `s < schedule` is false.
            // If all elements are >= schedule (e.g. have same time or later), it's 0.
            // If all elements are < schedule, it's queue.len().
            // For identical times, a new item is inserted at the first position where its time is not less than existing.
            // Example: [t=100(k_prev)]. Insert t=100(k_new). partition_point for k_new returns index of k_prev.
            // So k_new is inserted before k_prev.
            // So if key1 (t=100) is in, then key2 (t=100) is inserted, partition_point for key2 returns index of key1.
            // queue.insert(idx, key2) means key2 comes before key1.
            // So, in the queue (soonest is last): [key1, key2]
            let keys = timer_keys_from_queue(&state.queue);
            assert_eq!(keys, vec![key1, key2]);
        });
    }

    #[test]
    fn test_delete_timer_existing() {
        reset_state();
        let now = Instant::now();
        let key1 = insert_and_schedule_timer(timer_with_id(1), |_k| Schedule { key: _k, next_trigger: now + Duration::from_millis(100), repeat: DontRepeat });
        let key2 = insert_and_schedule_timer(timer_with_id(2), |_k| Schedule { key: _k, next_trigger: now + Duration::from_millis(200), repeat: DontRepeat });

        STATE.with_borrow(|state| {
            assert_eq!(state.timers.len(), 2);
            assert_eq!(state.queue.len(), 2);
        });

        let result = delete_timer(key1);
        assert!(result.is_ok());

        STATE.with_borrow(|state| {
            assert!(!state.timers.contains(key1));
            assert!(state.timers.contains(key2));
            assert_eq!(state.timers.len(), 1);

            let keys_in_queue = timer_keys_from_queue(&state.queue);
            assert!(!keys_in_queue.contains(&key1));
            assert!(keys_in_queue.contains(&key2));
            assert_eq!(state.queue.len(), 1);
        });
    }

    #[test]
    fn test_delete_timer_non_existent() {
        reset_state();
        let key1 = insert_and_schedule_timer(timer_with_id(1), |_k| Schedule { key: _k, next_trigger: Instant::now() + Duration::from_millis(100), repeat: DontRepeat });
        
        let non_existent_key = key1 + 100; // A key that's not in the slab

        let result = delete_timer(non_existent_key);
        assert!(result.is_err());
        match result.err().unwrap() {
            TriggeringError::TimerNotInQueue => {} // Expected error
            _ => panic!("Unexpected error type for non-existent timer deletion"),
        }

        // Ensure state is unchanged
        STATE.with_borrow(|state| {
            assert!(state.timers.contains(key1));
            assert_eq!(state.timers.len(), 1);
            assert_eq!(timer_keys_from_queue(&state.queue), vec![key1]);
        });
    }

    #[test]
    fn test_delete_timer_affects_only_specified_timer() {
        reset_state();
        let now = Instant::now();
        let key1 = insert_and_schedule_timer(timer_with_id(1), |_k| Schedule { key: _k, next_trigger: now + Duration::from_millis(100), repeat: DontRepeat });
        let key2 = insert_and_schedule_timer(timer_with_id(2), |_k| Schedule { key: _k, next_trigger: now + Duration::from_millis(200), repeat: DontRepeat });
        let key3 = insert_and_schedule_timer(timer_with_id(3), |_k| Schedule { key: _k, next_trigger: now + Duration::from_millis(50), repeat: DontRepeat }); // Soonest

        assert!(delete_timer(key2).is_ok());

        STATE.with_borrow(|state| {
            assert!(state.timers.contains(key1));
            assert!(!state.timers.contains(key2));
            assert!(state.timers.contains(key3));
            assert_eq!(state.timers.len(), 2);

            // Queue order: key1 (100ms), key3 (50ms)
            assert_eq!(timer_keys_from_queue(&state.queue), vec![key1, key3]);
        });
    }

    #[test]
    fn test_reschedule_timer_earlier() {
        reset_state();
        let now = Instant::now();
        let key1 = insert_and_schedule_timer(timer_with_id(1), |_k| Schedule { key: _k, next_trigger: now + Duration::from_millis(200), repeat: DontRepeat });
        let key2 = insert_and_schedule_timer(timer_with_id(2), |_k| Schedule { key: _k, next_trigger: now + Duration::from_millis(300), repeat: DontRepeat });

        // Reschedule key2 to be earlier than key1
        let new_schedule_key2 = Schedule { key: key2, next_trigger: now + Duration::from_millis(100), repeat: DontRepeat };
        assert!(reschedule_timer(key2, new_schedule_key2).is_ok());

        STATE.with_borrow(|state| {
            // Expected order: key1 (200ms), key2 (100ms)
            assert_eq!(timer_keys_from_queue(&state.queue), vec![key1, key2]);
            assert_eq!(state.queue.last().unwrap().next_trigger, new_schedule_key2.next_trigger);
        });
    }

    #[test]
    fn test_reschedule_timer_later() {
        reset_state();
        let now = Instant::now();
        let key1 = insert_and_schedule_timer(timer_with_id(1), |_k| Schedule { key: _k, next_trigger: now + Duration::from_millis(100), repeat: DontRepeat });
        let key2 = insert_and_schedule_timer(timer_with_id(2), |_k| Schedule { key: _k, next_trigger: now + Duration::from_millis(50), repeat: DontRepeat }); // key2 is soonest

        // Reschedule key2 to be later than key1
        let new_schedule_key2 = Schedule { key: key2, next_trigger: now + Duration::from_millis(200), repeat: DontRepeat };
        assert!(reschedule_timer(key2, new_schedule_key2).is_ok());

        STATE.with_borrow(|state| {
            // Expected order: key2 (200ms), key1 (100ms)
            assert_eq!(timer_keys_from_queue(&state.queue), vec![key2, key1]);
            assert_eq!(state.queue.iter().find(|s| s.key == key2).unwrap().next_trigger, new_schedule_key2.next_trigger);
        });
    }

    #[test]
    fn test_reschedule_timer_change_repeat_status() {
        reset_state();
        let now = Instant::now();
        let key1 = insert_and_schedule_timer(timer_with_id(1), |_k| Schedule { key: _k, next_trigger: now + Duration::from_millis(100), repeat: DontRepeat });
        
        // Change to repeating
        let new_repeat_schedule = Schedule { key: key1, next_trigger: now + Duration::from_millis(100), repeat: Every(Duration::from_secs(1)) };
        assert!(reschedule_timer(key1, new_repeat_schedule).is_ok());
        STATE.with_borrow(|state| {
            assert_eq!(state.queue.iter().find(|s| s.key == key1).unwrap().repeat, Every(Duration::from_secs(1)));
        });

        // Change back to non-repeating
        let new_dont_repeat_schedule = Schedule { key: key1, next_trigger: now + Duration::from_millis(100), repeat: DontRepeat };
        assert!(reschedule_timer(key1, new_dont_repeat_schedule).is_ok());
        STATE.with_borrow(|state| {
            assert_eq!(state.queue.iter().find(|s| s.key == key1).unwrap().repeat, DontRepeat);
        });
    }
    
    #[test]
    fn test_reschedule_timer_non_existent() {
        reset_state();
        let key1 = insert_and_schedule_timer(timer_with_id(1), |_k| Schedule { key: _k, next_trigger: Instant::now() + Duration::from_millis(100), repeat: DontRepeat });
        
        let non_existent_key = key1 + 100;
        let schedule_for_non_existent = Schedule { key: non_existent_key, next_trigger: Instant::now() + Duration::from_millis(50), repeat: DontRepeat };
        
        let result = reschedule_timer(non_existent_key, schedule_for_non_existent);
        assert!(result.is_err());
        match result.err().unwrap() {
            TriggeringError::TimerNotInQueue => {} // Expected
            _ => panic!("Unexpected error for rescheduling non-existent timer"),
        }
    }

    #[test]
    fn test_remove_timers_single() {
        reset_state();
        let now = Instant::now();
        let key1 = insert_and_schedule_timer(timer_with_id(10), |k| non_repeating_schedule(k, 100));
        let key2 = insert_and_schedule_timer(timer_with_id(20), |k| non_repeating_schedule(k, 200));
        let key3 = insert_and_schedule_timer(timer_with_id(30), |k| non_repeating_schedule(k, 50));

        // Remove timer with ID 20 (key2)
        remove_timers(|timer| get_timer_id(timer) == Some(20));

        STATE.with_borrow(|state| {
            assert!(state.timers.contains(key1));
            assert!(!state.timers.contains(key2)); // key2 should be removed
            assert!(state.timers.contains(key3));
            assert_eq!(state.timers.len(), 2);

            // Expected queue: key1 (100ms), key3 (50ms)
            assert_eq!(timer_keys_from_queue(&state.queue), vec![key1, key3]);
        });
    }

    #[test]
    fn test_remove_timers_multiple() {
        reset_state();
        let now = Instant::now();
        let key1 = insert_and_schedule_timer(timer_with_id(10), |k| non_repeating_schedule(k, 100));
        let key2 = insert_and_schedule_timer(timer_with_id(20), |k| non_repeating_schedule(k, 200));
        let key3 = insert_and_schedule_timer(timer_with_id(10), |k| non_repeating_schedule(k, 50)); // Another timer with ID 10

        // Remove timers with ID 10
        remove_timers(|timer| get_timer_id(timer) == Some(10));

        STATE.with_borrow(|state| {
            assert!(!state.timers.contains(key1)); // key1 (ID 10) removed
            assert!(state.timers.contains(key2));  // key2 (ID 20) remains
            assert!(!state.timers.contains(key3)); // key3 (ID 10) removed
            assert_eq!(state.timers.len(), 1);

            assert_eq!(timer_keys_from_queue(&state.queue), vec![key2]);
        });
    }

    #[test]
    fn test_remove_timers_none_matching() {
        reset_state();
        let now = Instant::now();
        let key1 = insert_and_schedule_timer(timer_with_id(10), |k| non_repeating_schedule(k, 100));
        let key2 = insert_and_schedule_timer(timer_with_id(20), |k| non_repeating_schedule(k, 200));

        // Predicate matches no existing timer IDs
        remove_timers(|timer| get_timer_id(timer) == Some(5));

        STATE.with_borrow(|state| {
            assert!(state.timers.contains(key1));
            assert!(state.timers.contains(key2));
            assert_eq!(state.timers.len(), 2);
            // Expected queue: key2 (200ms), key1 (100ms)
            assert_eq!(timer_keys_from_queue(&state.queue), vec![key2, key1]);
        });
    }

    #[test]
    fn test_remove_timers_all_matching() {
        reset_state();
        let now = Instant::now();
        insert_and_schedule_timer(timer_with_id(10), |k| non_repeating_schedule(k, 100));
        insert_and_schedule_timer(timer_with_id(10), |k| non_repeating_schedule(k, 200));

        // Predicate matches all timers (e.g., based on a common property or just true)
        remove_timers(|timer| get_timer_id(timer) == Some(10)); // Matches all based on ID

        STATE.with_borrow(|state| {
            assert_eq!(state.timers.len(), 0);
            assert_eq!(state.queue.len(), 0);
        });
    }

    #[test]
    fn test_reschedule_next_due_empty_queue() {
        reset_state();
        let mut closure_called = false;
        let result = reschedule_next_due_and_then(Instant::now(), |_timer| closure_called = true);
        assert!(result.is_none());
        assert!(!closure_called);
    }

    #[test]
    fn test_reschedule_next_due_non_due_timer() {
        reset_state();
        insert_and_schedule_timer(timer_with_id(1), |k| non_repeating_schedule(k, 1000)); // Due in 1 sec

        let mut closure_called = false;
        // 'now' is before the timer is due
        let result = reschedule_next_due_and_then(Instant::now(), |_timer| closure_called = true);
        assert!(result.is_none());
        assert!(!closure_called);
        STATE.with_borrow(|state| {
            assert_eq!(state.timers.len(), 1); // Timer should still be there
            assert_eq!(state.queue.len(), 1);
        });
    }

    #[test]
    fn test_reschedule_next_due_dont_repeat_timer() {
        reset_state();
        let key = insert_and_schedule_timer(timer_with_id(1), |k| non_repeating_schedule(k, 50)); // Due in 50ms

        std::thread::sleep(Duration::from_millis(100)); // Ensure timer is due

        let mut closure_called = false;
        let result = reschedule_next_due_and_then(Instant::now(), |timer| {
            closure_called = true;
            assert_eq!(get_timer_id(timer), Some(1));
        });
        
        assert!(result.is_some());
        assert!(closure_called);
        STATE.with_borrow(|state| {
            assert!(!state.timers.contains(key)); // Timer should be removed from slab
            assert!(state.queue.is_empty());    // Queue should be empty
        });
    }

    #[test]
    fn test_reschedule_next_due_every_timer() {
        reset_state();
        let interval = Duration::from_millis(200);
        let key = insert_and_schedule_timer(timer_with_id(1), |k| Schedule {
            key: k,
            next_trigger: Instant::now() + Duration::from_millis(50),
            repeat: Every(interval),
        });

        let initial_trigger_time = STATE.with_borrow(|state| state.queue.first().unwrap().next_trigger);
        
        std::thread::sleep(Duration::from_millis(100)); // Ensure timer is due
        let call_time = Instant::now();

        let mut closure_called = false;
        let result = reschedule_next_due_and_then(call_time, |timer| {
            closure_called = true;
            assert_eq!(get_timer_id(timer), Some(1));
        });

        assert!(result.is_some());
        assert!(closure_called);
        STATE.with_borrow(|state| {
            assert!(state.timers.contains(key)); // Timer should still be in slab
            assert_eq!(state.queue.len(), 1);    // Timer should be re-queued
            let new_schedule = state.queue.first().unwrap();
            assert_eq!(new_schedule.key, key);
            // Expected next trigger is call_time + interval
            assert!(new_schedule.next_trigger >= call_time + interval); 
            // Check it's reasonably close, allowing for small scheduling/execution overhead
            assert!(new_schedule.next_trigger <= call_time + interval + Duration::from_millis(50)); 
        });
    }

    #[test]
    fn test_reschedule_next_due_processes_only_one() {
        reset_state();
        let now = Instant::now();
        // Both timers will be due
        let key1 = insert_and_schedule_timer(timer_with_id(1), |k| non_repeating_schedule(k, 50));
        let key2 = insert_and_schedule_timer(timer_with_id(2), |k| non_repeating_schedule(k, 50)); 

        std::thread::sleep(Duration::from_millis(100)); // Ensure both are due
        
        let mut first_call_timer_id = 0;
        let result1 = reschedule_next_due_and_then(Instant::now(), |timer| {
            first_call_timer_id = get_timer_id(timer).unwrap_or(0);
        });
        assert!(result1.is_some());
        assert!(first_call_timer_id == 1 || first_call_timer_id == 2);

        STATE.with_borrow(|state| {
            assert_eq!(state.timers.len(), 1); // One timer processed and removed (DontRepeat)
            assert_eq!(state.queue.len(), 1);
        });

        let mut second_call_timer_id = 0;
        let result2 = reschedule_next_due_and_then(Instant::now(), |timer| {
            second_call_timer_id = get_timer_id(timer).unwrap_or(0);
        });
        assert!(result2.is_some());
        assert!(second_call_timer_id != 0 && second_call_timer_id != first_call_timer_id);
        
        STATE.with_borrow(|state| {
            assert_eq!(state.timers.len(), 0); // Second timer processed
            assert_eq!(state.queue.len(), 0);
        });
    }
}
