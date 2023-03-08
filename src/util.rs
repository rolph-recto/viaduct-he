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

/// generate a descending list of powers of 2
/// e.g. given n = 16, return [16, ]
pub fn gen_pow2_list(n: usize) -> Vec<usize> {
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

#[cfg(test)]
mod test {
    use super::gen_pow2_list;

    #[test]
    fn test_pow2() {
        assert!(gen_pow2_list(4) == vec![4, 2, 1])
    }

    #[test]
    fn test_pow2_2() {
        assert!(gen_pow2_list(1) == vec![1])
    }
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
}