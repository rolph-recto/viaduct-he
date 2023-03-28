use std::collections::HashMap;

pub struct NameGenerator {
    name_map: HashMap<String, usize>,
}

impl NameGenerator {
    pub fn new() -> Self {
        NameGenerator {
            name_map: HashMap::new(),
        }
    }

    /// register a name; returns true if it is fresh,
    /// false if it's already been registered
    pub fn register(&mut self, name: &str) -> bool {
        let fresh = !self.name_map.contains_key(name);
        if fresh {
            self.name_map.insert(String::from(name), 2);
        }

        fresh
    }

    pub fn get_fresh_name(&mut self, name: &str) -> String {
        if self.name_map.contains_key(name) {
            let n = self.name_map[name];
            let counter = self.name_map.get_mut(name).unwrap();
            *counter += 1;
            format!("{}_{}", name, n)

        } else {
            self.name_map.insert(String::from(name), 2);
            format!("{}_{}", name, 1)
        }
    }
}

impl Default for NameGenerator {
    fn default() -> Self {
        NameGenerator::new()
    }
}

pub fn get_nearest_pow2(n: usize) -> usize {
    let mut pow = 1;
    while pow < n {
        pow *= 2;
    }

    pow
}

/// generate a descending list of powers of 2
/// e.g. given n = 16, return [16, 8, 4, 2, 1]
pub fn descending_pow2_list(n: usize) -> Vec<usize> {
    // n must be a power of 2
    assert!(n >= 1 && n & (n - 1) == 0);

    let mut res = vec![];
    let mut cur = n;
    while cur >= 1 {
        res.push(cur);

        // logical shift on unsigned int
        cur >>= 1;
    }

    res
}

// get a list of factor pairs of n
// excludes the trivial factor pair (1, n)
pub fn get_factor_pairs(n: usize) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for i in (2..).take_while(|x| x * x <= n) {
        if n % i == 0 {
            let q = n / i;
            pairs.push((i, q));

            if i != q {
                pairs.push((q, i));
            }
        }
    }

    pairs
}

// return rotation steps for a list of dims (extent and blocksize pairs) to reduce
pub fn get_reduction_list(dims_to_fill: Vec<(usize, usize)>) -> Vec<isize> {
    let mut reduction_list: Vec<isize> = Vec::new();
    for (extent, block_size) in dims_to_fill {
        reduction_list.extend(
            descending_pow2_list(extent >> 1)
            .into_iter().rev().map(|x| (x * block_size) as isize)
        );
    }

    reduction_list
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fresh_names() {
        let mut namegen = NameGenerator::new();
        let name1 = namegen.get_fresh_name("test");
        let name2 = namegen.get_fresh_name("test");
        let name3 = namegen.get_fresh_name("test");

        assert_ne!(name1, name2);
        assert_ne!(name2, name3);
        assert_ne!(name1, name3);
    }

    #[test]
    fn test_pow2() {
        assert!(descending_pow2_list(4) == vec![4, 2, 1])
    }

    #[test]
    fn test_pow2_2() {
        assert!(descending_pow2_list(1) == vec![1])
    }

    #[test]
    fn test_factor_pairs() {
        assert_eq!(get_factor_pairs(4), vec![(2,2)]);
        assert_eq!(get_factor_pairs(10), vec![(2,5),(5,2)]);
        assert_eq!(get_factor_pairs(12), vec![(2,6),(6,2),(3,4),(4,3)]);
        assert_eq!(get_factor_pairs(16), vec![(2,8),(8,2),(4,4)]);
    }
}