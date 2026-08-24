//! Deterministic batch index generation.

use super::bad;
use crate::Result;

/// An iterator over deterministic index batches.
#[derive(Clone, Debug)]
pub struct BatchIter {
    order: Vec<usize>,
    at: usize,
    batch: usize,
    drop_last: bool,
}

impl BatchIter {
    pub fn new(
        len: usize,
        batch: usize,
        seed: u64,
        shuffle: bool,
        drop_last: bool,
    ) -> Result<Self> {
        if batch == 0 {
            return Err(bad("batch size must be nonzero"));
        }
        let mut order: Vec<usize> = (0..len).collect();
        if shuffle {
            for index in (1..len).rev() {
                let mut mixed = seed ^ (index as u64).wrapping_mul(0x9E3779B97F4A7C15);
                mixed ^= mixed >> 30;
                mixed = mixed.wrapping_mul(0xBF58476D1CE4E5B9);
                order.swap(index, (mixed as usize) % (index + 1));
            }
        }
        Ok(Self {
            order,
            at: 0,
            batch,
            drop_last,
        })
    }
}

impl Iterator for BatchIter {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.at >= self.order.len() {
            return None;
        }
        let end = (self.at + self.batch).min(self.order.len());
        if self.drop_last && end - self.at < self.batch {
            self.at = self.order.len();
            return None;
        }
        let output = self.order[self.at..end].to_vec();
        self.at = end;
        Some(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_order_is_seeded_and_drop_last_is_explicit() {
        assert_eq!(
            BatchIter::new(5, 2, 7, true, false)
                .unwrap()
                .collect::<Vec<_>>(),
            BatchIter::new(5, 2, 7, true, false)
                .unwrap()
                .collect::<Vec<_>>()
        );
        assert_eq!(
            BatchIter::new(5, 2, 0, false, false)
                .unwrap()
                .collect::<Vec<_>>(),
            vec![vec![0, 1], vec![2, 3], vec![4]]
        );
        assert_eq!(BatchIter::new(5, 2, 0, false, true).unwrap().count(), 2);
        assert!(BatchIter::new(1, 0, 0, false, false).is_err());
    }
}
