use std::collections::HashMap;

pub struct NameGenerator {
    name_map: HashMap<String, usize>
}

impl NameGenerator {
    pub fn new() -> Self {
        NameGenerator { name_map: HashMap::new() }
    }

    pub fn get_fresh_name(&mut self, name: &str) -> String {
        if self.name_map.contains_key(name) {
            let n = self.name_map[name];
            format!("{}_{}", name, n)
            
        } else {
            self.name_map.insert(String::from(name), 2);
            return format!("{}_{}", name, 1)
        }
    }
}

impl Default for NameGenerator {
    fn default() -> Self {
        NameGenerator::new()
    }
}