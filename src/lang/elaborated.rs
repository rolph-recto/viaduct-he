use std::{collections::{HashMap, HashSet}, fmt::Display};

use indexmap::IndexMap;

use crate::util::NameGenerator;

use super::{ArrayName, Shape, SourceExpr, SourceProgram, OUTPUT_EXPR_NAME, IndexingId, ArrayType};

/// like SourceProgram, but with uniquely named indexing sites
pub struct ElaboratedProgram {
    pub input_map: IndexMap<ArrayName, (Shape, ArrayType)>,
    pub expr_map: IndexMap<IndexingId, SourceExpr>,

    // map from new names to old names
    pub rename_map: HashMap<IndexingId, ArrayName>,
}

impl ElaboratedProgram {
    // get a dependency map between let-bound expressions
    pub fn get_expr_dependency_map(&self) -> HashMap<IndexingId, HashSet<IndexingId>> {
        let mut dependency_map: HashMap<ArrayName, HashSet<ArrayName>> = HashMap::new();
        for (array, expr) in self.expr_map.iter() {
            let expr_indexed: HashSet<ArrayName> =
                expr.get_indexed_arrays()
                .into_iter()
                .filter(|indexed| self.is_expr(indexed))
                .collect();

            dependency_map.insert(array.clone(), expr_indexed);
        }

        dependency_map
    }

    pub fn is_expr(&self, indexing_id: &IndexingId) -> bool {
        self.expr_map.contains_key(indexing_id)
    }

    pub fn is_input(&self, indexing_id: &IndexingId) -> bool {
        self.input_map.contains_key(&self.rename_map[indexing_id])
    }

    /// the default inline set contains all indexing ids
    pub fn get_default_inline_set(&self) -> HashSet<IndexingId> {
        self.expr_map.iter().map(|(id, _)| id.clone()).collect()
    }

    /// the default array group map just contains mappings for indexing ids for inputs
    pub fn get_default_array_group_map(&self) -> HashMap<IndexingId, ArrayName> {
        let mut array_group_map: HashMap<IndexingId, ArrayName> = HashMap::new();

        // indexing ids for the same input array must be mapped together
        for (_, expr) in self.expr_map.iter() {
            for indexing_id in expr.get_indexed_arrays() {
                if self.is_input(&indexing_id) {
                    let array = self.rename_map.get(&indexing_id).unwrap().clone();
                    array_group_map.insert(indexing_id, array);

                } else {
                    array_group_map.insert(indexing_id.clone(), indexing_id);
                }
            }
        }

        array_group_map
    }
}

impl Display for ElaboratedProgram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.input_map.iter().try_for_each(|(array, shape)| {
            write!(f, "input {}: {:?}\n", array, shape)
        })?;

        self.expr_map.iter().try_for_each(|(id, expr)| {
            write!(f, "let {} = {}\n", id, expr)
        })?;

        Ok(())
    }
}

// the elaborator uniqifies indexing sites
pub struct Elaborator {
    name_generator: NameGenerator
}

impl Elaborator {
    pub fn new() -> Self {
        Self { name_generator: NameGenerator::new() }
    }

    pub fn run(mut self, program: SourceProgram) -> ElaboratedProgram {
        let mut expr_map: IndexMap<IndexingId, SourceExpr> = IndexMap::new();

        let output_name = String::from(OUTPUT_EXPR_NAME);
        let mut rename_map: HashMap<IndexingId, ArrayName> = HashMap::new();
        rename_map.insert(output_name.clone(), output_name.clone());

        let mut worklist: Vec<IndexingId> = vec![output_name];
        while !worklist.is_empty() {
            let cur: IndexingId = worklist.pop().unwrap();
            let cur_old_name: &ArrayName = rename_map.get(&cur).unwrap();
            if program.is_expr(cur_old_name) {
                let expr = program.get_expr(cur_old_name).unwrap();
                let (new_expr, expr_renames) = self.elaborate_expr(expr.clone());
                expr_map.insert(cur.clone(), new_expr);

                for (k, v) in expr_renames {
                    worklist.push(k.clone());
                    rename_map.insert(k, v);
                }

            }
        }

        // elaboration runs backwards from the output expr,
        // so reverse the insertion order to get the program order
        expr_map.reverse();
        ElaboratedProgram { input_map: program.input_map, expr_map, rename_map }
    }
    
    pub fn elaborate_expr(&mut self, expr: SourceExpr) -> (SourceExpr, HashMap<ArrayName, ArrayName>) {
        match expr {
            SourceExpr::For(var, extent, body) => {
                let (new_body, body_renames) = self.elaborate_expr(*body);
                (SourceExpr::For(var, extent, Box::new(new_body)), body_renames)
            },

            SourceExpr::Reduce(op, body) => {
                let (new_body, body_renames) = self.elaborate_expr(*body);
                (SourceExpr::Reduce(op, Box::new(new_body)), body_renames)
            },
            
            SourceExpr::ExprOp(op, expr1, expr2) => {
                let (new_expr1, mut expr1_renames) = self.elaborate_expr(*expr1);
                let (new_expr2, mut expr2_renames) = self.elaborate_expr(*expr2);

                for (k, v)in expr2_renames {
                    expr1_renames.insert(k,v);
                }

                (SourceExpr::ExprOp(op, Box::new(new_expr1), Box::new(new_expr2)), expr1_renames)
            },

            SourceExpr::Literal(_) =>
                (expr, HashMap::new()),

            SourceExpr::Indexing(array, index) => {
                let new_array = self.name_generator.get_fresh_name(&array);
                let rename = HashMap::from([(new_array.clone(), array)]);
                (SourceExpr::Indexing(new_array, index), rename)
            }
        }
    }
}