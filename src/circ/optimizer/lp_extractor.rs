use std::collections::{HashMap, HashSet};
use crate::circ::optimizer::*;
use good_lp::{*, solvers::coin_cbc::CoinCbcProblem};

pub(crate) struct OpSizeFunction;

impl LpCostFunction<HEOptCircuit, HEData> for OpSizeFunction {
    fn node_cost(&mut self, egraph: &HEGraph, id: Id, enode: &HEOptCircuit) -> f64 {
        match enode {
            HEOptCircuit::Num(_) => 0.1,

            HEOptCircuit::Add([id1, id2]) => {
                let d1 = egraph[*id1].data.muldepth;
                let d2 = egraph[*id2].data.muldepth;
                let muldepth = max(d1, d2) + 1;
                6.0 * (muldepth as f64)
            },

            HEOptCircuit::Sub([id1, id2]) => {
                let d1 = egraph[*id1].data.muldepth;
                let d2 = egraph[*id2].data.muldepth;
                let muldepth = max(d1, d2) + 1;
                6.0 * (muldepth as f64)
            },

            HEOptCircuit::Mul([id1, id2]) => {
                let d1 = egraph[*id1].data.muldepth;
                let d2 = egraph[*id2].data.muldepth;
                let muldepth = max(d1, d2) + 1;
                15.0 * (muldepth as f64)
            },

            HEOptCircuit::Rot([id1, id2]) => {
                let d1 = egraph[*id1].data.muldepth;
                let d2 = egraph[*id2].data.muldepth;
                let muldepth = max(d1, d2) + 1;
                0.1 * (muldepth as f64)
            },

            HEOptCircuit::CiphertextRef(_) => 0.1,

            HEOptCircuit::PlaintextRef(_) => 0.1,
        }
    }
}

pub struct HEExtractor<'a> {
    egraph: &'a HEGraph,
    class_vars: HashMap<Id, ClassVar>,
    model: CoinCbcProblem,
    root: Id,
}

struct ClassVar {
    is_active: Variable,
    latency: Variable,
    muldepth: Variable,
    node_vars: Vec<Variable>,
}

impl<'a> HEExtractor<'a> {
    pub fn new(egraph: &'a HEGraph, root: Id) -> Self {
        let mut vars = variables!();
        let class_vars: HashMap<Id, ClassVar> =
            egraph.classes().map(|cls| {
                let cls_var =
                    ClassVar {
                        is_active: vars.add(variable().binary()),
                        latency: vars.add(variable().min(0.0)),
                        muldepth: vars.add(variable().integer().min(0)),
                        node_vars:
                            cls.nodes.iter()
                            .map(|node| vars.add(variable().binary()))
                            .collect(),
                    };
                (cls.id, cls_var)
            }).collect();

        let root_var = class_vars.get(&root).unwrap();
        
        let mut cycles: HashSet<(Id, usize)> = Default::default();
        find_cycles(egraph, |id, i| {
            cycles.insert((id, i));
        });

        let mut model =
            vars
            .minimise(root_var.latency + root_var.muldepth)
            .using(default_solver);

        // big-M coefficient used for conditional constraints
        let big_m = 10000000;

        // root node has to be active
        model.add_constraint(constraint!(root_var.is_active == 1));

        for (&id, cls) in class_vars.iter() {
            let node_sum: Expression =  cls.node_vars.iter().map(Expression::from_other_affine).sum();
            model.add_constraint(constraint!(node_sum - cls.is_active == 0));

            for (i, (node, &is_node_active)) in egraph[id].iter().zip(&cls.node_vars).enumerate() {
                // don't activate node if it is part of a cycle
                if cycles.contains(&(id, i)) {
                    model.add_constraint(constraint!(is_node_active == 0));
                    continue;
                }

                let is_node_mul =
                    match node {
                        HEOptCircuit::Mul(_) => true,
                        _ => false,
                    } && egraph[id].data.constval.is_none();

                let mut are_children_const = true;
                for child in node.children() {
                    let is_child_active = class_vars[child].is_active;
                    let child_muldepth = class_vars[child].muldepth;

                    // node active implies child active, encoded as:
                    // is_node_active <= is_child_active
                    // is_node_active - is_child_active <= 0
                    model.add_constraint(constraint!(is_node_active - is_child_active <= 0));

                    // conditional constraint:
                    // if node is active, then class muldepth is max(child_muldepth)
                    // so child muldepth lower bounds class muldepth if node is active
                    // encode using a "big-M" constraint
                    if is_node_mul {
                        model.add_constraint(
                            constraint!(big_m * (1 - is_node_active) >= child_muldepth + 1 - cls.muldepth)
                        );

                    } else {
                        model.add_constraint(
                            constraint!(big_m * (1 - is_node_active) >= child_muldepth - cls.muldepth)
                        );
                    }

                    are_children_const = are_children_const && egraph[*child].data.constval.is_some();
                }

                // TODO must check that *children* are constant
                let latency =
                    match node {
                        HEOptCircuit::Num(_) =>
                            NUM_LATENCY,

                        HEOptCircuit::Add(_) =>
                            if are_children_const {
                                ADD_PLAIN_LATENCY
                            } else {
                                ADD_LATENCY
                            },

                        HEOptCircuit::Sub(_) =>
                            if are_children_const {
                                SUB_PLAIN_LATENCY
                            } else {
                                SUB_LATENCY
                            },

                        HEOptCircuit::Mul(_) =>
                            if are_children_const {
                                MUL_PLAIN_LATENCY
                            } else {
                                MUL_LATENCY
                            },

                        HEOptCircuit::Rot(_) => 
                            ROT_LATENCY,

                        HEOptCircuit::CiphertextRef(_) |
                        HEOptCircuit::PlaintextRef(_) =>
                            SYM_LATENCY,
                    } as f64;

                    model.add_constraint(
                        constraint!(big_m * (1 - is_node_active) >= Expression::from(latency) - cls.latency)
                    );
            }
        }

        Self { egraph, model, class_vars, root }
    }

    pub fn solve(self) -> Result<RecExpr<HEOptCircuit>, ResolutionError> {
        let solution = self.model.solve()?;
        let mut expr: RecExpr<HEOptCircuit> = RecExpr::default();
    
        let mut worklist: Vec<Id> = Vec::from([self.egraph.find(self.root)]);
        let mut ids: HashMap<Id, Id> = HashMap::default();

        while let Some(&id) = worklist.last() {
            if ids.contains_key(&id) {
                worklist.pop();
                continue;
            }

            let class_var = &self.class_vars[&id];
            assert!(solution.value(class_var.is_active) > 0.0);

            let node_idx =
                class_var.node_vars.iter()
                .position(|&n| solution.value(n) > 0.0)
                .unwrap();
            let node = &self.egraph[id].nodes[node_idx];

            if node.all(|child| ids.contains_key(&child)) {
                let new_id =
                    expr.add(
                        node.clone().map_children(|i| ids[&self.egraph.find(i)])
                    );
                ids.insert(id, new_id);
                worklist.pop();

            } else {
                worklist.extend_from_slice(node.children())
            }
        }

        Ok(expr)
    }
}

// copied from egg's lp_extract module
fn find_cycles<L, N>(egraph: &EGraph<L, N>, mut f: impl FnMut(Id, usize))
where
    L: Language,
    N: Analysis<L>,
{
    enum Color {
        White,
        Gray,
        Black,
    }
    type Enter = bool;

    let mut color: HashMap<Id, Color> = egraph.classes().map(|c| (c.id, Color::White)).collect();
    let mut stack: Vec<(Enter, Id)> = egraph.classes().map(|c| (true, c.id)).collect();

    while let Some((enter, id)) = stack.pop() {
        if enter {
            *color.get_mut(&id).unwrap() = Color::Gray;
            stack.push((false, id));
            for (i, node) in egraph[id].iter().enumerate() {
                for child in node.children() {
                    match &color[child] {
                        Color::White => stack.push((true, *child)),
                        Color::Gray => f(id, i),
                        Color::Black => (),
                    }
                }
            }
        } else {
            *color.get_mut(&id).unwrap() = Color::Black;
        }
    }
}