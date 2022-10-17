use std::collections::{HashMap, HashSet};
use crate::circ::optimizer::*;
use coin_cbc::{Model, Col, Sense};
use good_lp::{*, solvers::coin_cbc::CoinCbcProblem};

pub struct OpSizeFunction { pub latency: HELatencyModel }

impl LpCostFunction<HEOptCircuit, HEData> for OpSizeFunction {
    fn node_cost(&mut self, egraph: &HEGraph, _: Id, enode: &HEOptCircuit) -> f64 {
        let child_muldepth =
            enode.children().iter().fold(0, |acc, child| {
                max(acc, egraph[*child].data.muldepth)
            });

        let is_plainop =
            enode.children().iter().any(|child| {
                egraph[*child].data.constval.is_some()
            });

        let mut muldepth = child_muldepth;
        let latency = 
            match enode {
                HEOptCircuit::Num(_) => self.latency.num,

                HEOptCircuit::Add(_) => {
                    if is_plainop {
                        self.latency.add_plain
                    } else {
                        self.latency.add
                    }
                },

                HEOptCircuit::Sub(_) => {
                    if is_plainop {
                        self.latency.sub_plain
                    } else {
                        self.latency.sub
                    }
                },

                HEOptCircuit::Mul(_) => {
                    if is_plainop {
                        self.latency.mul_plain
                    } else {
                        muldepth += 1;
                        self.latency.mul
                    }
                },

                HEOptCircuit::Rot(_) => self.latency.rot,

                HEOptCircuit::CiphertextRef(_) => self.latency.sym,

                HEOptCircuit::PlaintextRef(_) => self.latency.sym,
            };

        ((muldepth+1) as f64) * latency
    }
}

/*
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
                        muldepth: vars.add(variable().min(0.0)),
                        node_vars:
                            cls.nodes.iter()
                            .map(|_| vars.add(variable().binary()))
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
        let big_m = 1000000000.0;

        // root node has to be active
        model.add_constraint(constraint!(root_var.is_active - 1 == 0));

        for (&id, cls) in class_vars.iter() {
            let node_sum: Expression =  cls.node_vars.iter().map(|&var| Expression::from(var)).sum();
            model.add_constraint(constraint!(node_sum - cls.is_active == 0));

            for (i, (node, &is_node_active)) in egraph[id].iter().zip(&cls.node_vars).enumerate() {
                // don't activate node if it is part of a cycle
                if cycles.contains(&(id, i)) {
                    model.add_constraint(constraint!(is_node_active == 0));
                    // println!("cycle! root: {} cur: {} node: {}", root, id, i);
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
                            constraint!((-big_m * is_node_active) - child_muldepth + cls.muldepth >= 1.0 - big_m)
                        );

                    } else {
                        model.add_constraint(
                            constraint!((-big_m * is_node_active) - child_muldepth + cls.muldepth >= -big_m)
                        );
                    }

                    are_children_const = are_children_const && egraph[*child].data.constval.is_some();
                }

                let op_latency =
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
                        constraint!((-big_m * is_node_active) - Expression::from(op_latency) + cls.latency >= -big_m)
                    );
            }
        }

        Self { egraph, model, class_vars, root }
    }

    // solve LP model and extract solution 
    // very similar to egg's LpExtractor::solve_multiple
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
*/

/*
pub struct HEExtractor<'a> {
    egraph: &'a HEGraph,
    model: Model,
    vars: HashMap<Id, ClassVars>,
    root: Id,
}

struct ClassVars {
    active: Col,
    muldepth: Col,
    order: Col,
    nodes: Vec<Col>,
}

impl<'a> HEExtractor<'a> {
    /// Create an [`LpExtractor`] using costs from the given [`LpCostFunction`].
    /// See those docs for details.
    pub fn new(egraph: &'a HEGraph, root: Id) -> Self {
        let max_order = egraph.total_number_of_nodes() as f64 * 10.0;
        let big_m = 1000000000.0;

        let mut model = Model::default();

        let vars: HashMap<Id, ClassVars> = egraph
            .classes()
            .map(|class| {
                let cvars = ClassVars {
                    active: model.add_binary(),
                    muldepth: model.add_col(),
                    order: model.add_col(),
                    nodes: class.nodes.iter().map(|_| model.add_binary()).collect(),
                };
                model.set_col_upper(cvars.order, max_order);
                (class.id, cvars)
            })
            .collect();

        let mut cycles: HashSet<(Id, usize)> = Default::default();
        find_cycles(egraph, |id, i| {
            cycles.insert((id, i));
        });

        for (&id, class) in &vars {
            // class active == some node active
            // sum(for node_active in class) == class_active
            let row = model.add_row();
            model.set_row_equal(row, 0.0);
            model.set_weight(row, class.active, -1.0);
            for &node_active in &class.nodes {
                model.set_weight(row, node_active, 1.0);
            }

            for (i, (node, &node_active)) in egraph[id].iter().zip(&class.nodes).enumerate() {
                if cycles.contains(&(id, i)) {
                    model.set_col_upper(node_active, 0.0);
                    model.set_col_lower(node_active, 0.0);
                    continue;
                }

                let is_node_mul =
                    match node {
                        HEOptCircuit::Mul(_) => true,
                        _ => false,
                    } && egraph[id].data.constval.is_none();

                for child in node.children() {
                    let child_active = vars[child].active;
                    let child_muldepth = vars[child].muldepth;

                    // node active implies child active, encoded as:
                    //   node_active <= child_active
                    //   node_active - child_active <= 0
                    let row = model.add_row();
                    model.set_row_upper(row, 0.0);
                    model.set_weight(row, node_active, 1.0);
                    model.set_weight(row, child_active, -1.0);

                    // conditional constraint:
                    // if node is active, then class muldepth is max(child_muldepth)
                    // so child muldepth lower bounds class muldepth if node is active
                    // encode using a "big-M" constraint
                    if is_node_mul {
                        let row = model.add_row();
                        model.set_row_lower(row, 1.0 - big_m);
                        model.set_weight(row, node_active, -big_m);
                        model.set_weight(row, child_muldepth, -1.0);
                        model.set_weight(row, class.muldepth, 1.0);
                    } else {
                        let row = model.add_row();
                        model.set_row_lower(row, -big_m);
                        model.set_weight(row, node_active, -big_m);
                        model.set_weight(row, child_muldepth, -1.0);
                        model.set_weight(row, class.muldepth, 1.0);
                    }
                }
            }
        }

        model.set_obj_sense(Sense::Minimize);
        model.set_obj_coeff(vars[&egraph.find(root)].muldepth, 10.0);

        for class in egraph.classes() {
            for (node, &node_active) in class.iter().zip(&vars[&class.id].nodes) {
                let mut are_children_const = true;
                for child in node.children() {
                    are_children_const = are_children_const && egraph[egraph.find(*child)].data.constval.is_some();
                }

                let op_latency =
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

                model.set_obj_coeff(node_active, op_latency);
            }
        }

        dbg!(max_order);

        Self {
            egraph,
            model,
            vars,
            root: egraph.find(root),
        }
    }

    /// Set the cbc timeout in seconds.
    pub fn timeout(&mut self, seconds: f64) -> &mut Self {
        self.model.set_parameter("seconds", &seconds.to_string());
        self
    }

    /// Extract a single rooted term.
    pub fn solve(&mut self) -> RecExpr<HEOptCircuit> {
        for class in self.vars.values() {
            self.model.set_binary(class.active);
        }

        self.model.set_col_lower(self.vars[&self.root].active, 1.0);

        let solution = self.model.solve();
        log::info!(
            "CBC status {:?}, {:?}",
            solution.raw().status(),
            solution.raw().secondary_status()
        );

        let mut todo: Vec<Id> = Vec::from([self.egraph.find(self.root)]);
        let mut expr = RecExpr::default();
        // converts e-class ids to e-node ids
        let mut ids: HashMap<Id, Id> = HashMap::default();

        while let Some(&id) = todo.last() {
            if ids.contains_key(&id) {
                todo.pop();
                continue;
            }
            let v = &self.vars[&id];
            assert!(solution.col(v.active) > 0.0);
            let node_idx = v.nodes.iter().position(|&n| solution.col(n) > 0.0).unwrap();
            let node = &self.egraph[id].nodes[node_idx];
            if node.all(|child| ids.contains_key(&child)) {
                let new_id = expr.add(node.clone().map_children(|i| ids[&self.egraph.find(i)]));
                ids.insert(id, new_id);
                todo.pop();
            } else {
                todo.extend_from_slice(node.children())
            }
        }

        assert!(expr.is_dag(), "LpExtract found a cyclic term!: {:?}", expr);
        expr
    }
}
*/


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
