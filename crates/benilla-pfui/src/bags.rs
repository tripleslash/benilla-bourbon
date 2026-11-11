//! Paced atomic bag operations. Whole-stack consolidation uses a swap packet because the
//! 1.12 server's SplitItem rejects count == source count (Tortoise Player.cpp:13044).

use benilla_ui::script::ContainerMove;
use std::collections::VecDeque;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Slot {
    pub guid: u64,
    pub id: u32,
    pub count: u32,
    pub max_stack: u32,
    pub locked: bool,
}

#[derive(Clone, Copy)]
struct Expected {
    guid: Option<u64>,
    id: u32,
    count: u32,
}

impl Expected {
    fn exact(slot: Slot) -> Self {
        Self {
            guid: Some(slot.guid),
            id: slot.id,
            count: slot.count,
        }
    }

    fn stack(slot: Slot, id: u32, count: u32) -> Self {
        if count == 0 {
            Self::exact(Slot::default())
        } else {
            Self {
                guid: (slot.guid != 0).then_some(slot.guid),
                id,
                count,
            }
        }
    }

    fn matches(self, slot: Slot) -> bool {
        self.guid.is_none_or(|guid| guid == slot.guid)
            && self.id == slot.id
            && self.count == slot.count
    }
}

struct Waiting {
    original: ContainerMove,
    expected: [Expected; 2],
    guids: [u64; 2],
    deadline: f64,
}

#[derive(Default)]
pub struct Executor {
    queue: VecDeque<ContainerMove>,
    waiting: Option<Waiting>,
    blocked_since: Option<f64>,
}

impl Executor {
    pub fn enqueue(
        &mut self,
        moves: impl IntoIterator<Item = ContainerMove>,
    ) -> Result<(), String> {
        for movement in moves {
            if self.queue.len() >= 512 {
                self.clear();
                return Err("pfUI bag operation queue exceeded 512 operations".into());
            }
            self.queue.push_back(movement);
        }
        Ok(())
    }

    pub fn busy(&self) -> bool {
        self.waiting.is_some() || !self.queue.is_empty()
    }

    pub fn pending_guids(&self) -> Option<[u64; 2]> {
        self.waiting.as_ref().map(|waiting| waiting.guids)
    }

    pub fn clear(&mut self) {
        self.queue.clear();
        self.waiting = None;
        self.blocked_since = None;
    }

    /// Ignore failures naming another outstanding stock operation.
    pub fn failed(&mut self, guid: u64) -> bool {
        if self
            .waiting
            .as_ref()
            .is_some_and(|w| guid == 0 || w.guids.contains(&guid))
        {
            self.clear();
            true
        } else {
            false
        }
    }

    /// At most one send; a following operation sees both confirmed endpoints, never a virtual
    /// or partially received stack. A cold slot/lock gets the same ten-second bound as an ACK.
    pub fn next(
        &mut self,
        now: f64,
        read: impl Fn(i64, u32) -> Option<Slot>,
    ) -> Result<Option<ContainerMove>, String> {
        let result = self.advance(now, read);
        if result.is_err() {
            self.clear();
        }
        result
    }

    fn advance(
        &mut self,
        now: f64,
        read: impl Fn(i64, u32) -> Option<Slot>,
    ) -> Result<Option<ContainerMove>, String> {
        if let Some(waiting) = &self.waiting {
            let mv = &waiting.original;
            let confirmed = read(mv.src_bag, mv.src_slot)
                .zip(read(mv.dst_bag, mv.dst_slot))
                .is_some_and(|(src, dst)| {
                    waiting.expected[0].matches(src) && waiting.expected[1].matches(dst)
                });
            if !confirmed {
                if now >= waiting.deadline {
                    return Err("pfUI bag operation timed out waiting for the server".into());
                }
                return Ok(None);
            }
            self.waiting = None;
        }
        let Some(mv) = self.queue.front().cloned() else {
            return Ok(None);
        };
        let pair = read(mv.src_bag, mv.src_slot).zip(read(mv.dst_bag, mv.dst_slot));
        let Some((src, dst)) = pair.filter(|(src, dst)| !src.locked && !dst.locked) else {
            let since = *self.blocked_since.get_or_insert(now);
            if now - since >= 10. {
                return Err(
                    "pfUI bag operation timed out waiting for inventory data or locks".into(),
                );
            }
            return Ok(None);
        };
        self.blocked_since = None;
        if src.guid == 0 || src.count == 0 {
            return Err("pfUI bag operation source changed or disappeared".into());
        }
        let mut wire = mv.clone();
        let expected = if let Some(count) = mv.count {
            if count == 0 || count > 255 || count > src.count {
                return Err("pfUI bag move count exceeds the current stack".into());
            }
            if dst.guid != 0
                && (src.id != dst.id || count > src.max_stack.saturating_sub(dst.count))
            {
                return Err(
                    "pfUI bag move destination cannot hold the whole requested count".into(),
                );
            }
            if count == src.count {
                wire.count = None;
            }
            [
                Expected::stack(src, src.id, src.count - count),
                Expected::stack(dst, src.id, dst.count + count),
            ]
        } else if dst.guid != 0 && src.id == dst.id && dst.count < src.max_stack {
            let merged = (src.max_stack - dst.count).min(src.count);
            [
                Expected::stack(src, src.id, src.count - merged),
                Expected::stack(dst, src.id, dst.count + merged),
            ]
        } else {
            [Expected::exact(dst), Expected::exact(src)]
        };
        self.queue.pop_front();
        self.waiting = Some(Waiting {
            original: mv,
            expected,
            guids: [src.guid, dst.guid],
            deadline: now + 10.,
        });
        Ok(Some(wire))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    fn movement(src: u32, dst: u32, count: Option<u32>) -> ContainerMove {
        ContainerMove {
            src_bag: 0,
            src_slot: src,
            dst_bag: 0,
            dst_slot: dst,
            count,
        }
    }
    fn stack(guid: u64, count: u32) -> Slot {
        Slot {
            guid,
            id: 117,
            count,
            max_stack: 20,
            locked: false,
        }
    }

    #[test]
    fn consolidation_rechecks_counts_and_waits_for_both_descriptor_updates() {
        let slots = RefCell::new([Slot::default(), stack(1, 18), stack(2, 5), stack(3, 1)]);
        let read = |_, slot| slots.borrow().get(slot as usize).copied();
        let mut worker = Executor::default();
        worker
            .enqueue([movement(2, 1, Some(2)), movement(2, 3, Some(3))])
            .unwrap();
        assert_eq!(
            worker.next(0., read).unwrap(),
            Some(movement(2, 1, Some(2)))
        );
        assert!(worker.next(0.1, read).unwrap().is_none());
        slots.borrow_mut()[1].count = 20;
        assert!(
            worker.next(0.2, read).unwrap().is_none(),
            "one endpoint is not an ACK"
        );
        slots.borrow_mut()[2].count = 3;
        assert_eq!(
            worker.next(0.3, read).unwrap(),
            Some(movement(2, 3, None)),
            "the remaining full stack must merge, not split"
        );
        slots.borrow_mut()[2] = Slot::default();
        slots.borrow_mut()[3].count = 4;
        assert!(worker.next(0.4, read).unwrap().is_none());
        assert!(!worker.busy());
    }

    #[test]
    fn placement_uses_confirmed_slots_and_preserves_each_instance() {
        let slots = RefCell::new([
            stack(10, 1),
            Slot {
                guid: 20,
                id: 6948,
                count: 1,
                max_stack: 1,
                locked: false,
            },
            Slot::default(),
        ]);
        let read = |_, slot| slots.borrow().get(slot as usize).copied();
        let mut worker = Executor::default();
        worker
            .enqueue([movement(0, 1, None), movement(0, 2, None)])
            .unwrap();
        assert_eq!(worker.next(0., read).unwrap(), Some(movement(0, 1, None)));
        slots.borrow_mut().swap(0, 1);
        assert_eq!(worker.next(0.1, read).unwrap(), Some(movement(0, 2, None)));
        slots.borrow_mut().swap(0, 2);
        worker.next(0.2, read).unwrap();
        assert!(!worker.busy());
        assert_eq!(slots.borrow()[2].id, 6948);
    }

    #[test]
    fn rejected_overflow_timeout_and_failure_stop_the_remaining_batch() {
        let mut worker = Executor::default();
        worker.enqueue([movement(1, 2, Some(5))]).unwrap();
        assert!(worker
            .next(0., |_, slot| Some(if slot == 1 {
                stack(1, 5)
            } else {
                stack(2, 19)
            }))
            .is_err());
        assert!(!worker.busy());
        worker
            .enqueue([movement(1, 2, Some(1)), movement(2, 3, None)])
            .unwrap();
        let read = |_, slot| {
            Some(if slot == 1 {
                stack(1, 5)
            } else {
                Slot::default()
            })
        };
        assert!(worker.next(1., read).unwrap().is_some());
        assert!(!worker.failed(999));
        assert!(worker.next(11., read).is_err());
        assert!(!worker.busy());
        worker.enqueue([movement(1, 2, Some(1))]).unwrap();
        worker.next(12., read).unwrap();
        assert!(worker.failed(1));
        assert!(!worker.busy());
    }
}
