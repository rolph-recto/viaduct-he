use clap::ValueEnum;
use egg::*;
use log::*;
use std::{
    cmp::max,
    collections::{HashSet, HashMap},
    time::*,
};

use crate::circ::optimizer::cost::{HECostFunction, HELatencyModel, HELpCostFunction};

use self::cost::HECostContext;

use super::VarName;

mod greedy_extractor;
pub mod lp_extractor;
mod dijkstra_extractor;
pub mod cost;

define_language! {
    /// The language used by egg e-graph engine.
    pub enum HEOptCircuit {
        Literal(isize),
        "+" = Add([Id; 2]),
        "-" = Sub([Id; 2]),
        "*" = Mul([Id; 2]),
        "rot" = Rot([Id; 2]),
        "sumvec" = SumVectors([Id; 2]),
        "prodvec" = ProductVectors([Id; 2]),
        CiphertextVar(Symbol),
        PlaintextVar(Symbol),
        IndexVar(Symbol),
        FunctionVar(Symbol, Vec<Id>),
    }
}

#[derive(Clone, ValueEnum)]
pub enum ExtractorType {
    Greedy,
    LP,
}

pub type HEGraph = egg::EGraph<HEOptCircuit, HEData>;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum HEOptNodeType {
    Cipher,
    Plain
}

impl HEOptNodeType {
    // combine the types of operands
    fn ops(&self, other: &Self) -> Self {
        match (self, other) {
            (HEOptNodeType::Cipher, HEOptNodeType::Cipher) |
            (HEOptNodeType::Cipher, HEOptNodeType::Plain) |
            (HEOptNodeType::Plain, HEOptNodeType::Cipher) =>
                HEOptNodeType::Cipher,

            (HEOptNodeType::Plain, HEOptNodeType::Plain) =>
                HEOptNodeType::Plain
        }
    }

    // pick the least type, where the ordering is
    // Plain <= CipherPlain <= Cipher
    fn merge(&self, other: &Self) -> Self {
        match (self, other) {
            (HEOptNodeType::Cipher, HEOptNodeType::Cipher) =>
                HEOptNodeType::Cipher,

            (HEOptNodeType::Cipher, HEOptNodeType::Plain) |
            (HEOptNodeType::Plain, HEOptNodeType::Cipher) |
            (HEOptNodeType::Plain, HEOptNodeType::Plain) =>
                HEOptNodeType::Plain
        }
    }
}

impl Default for HEOptNodeType {
    fn default() -> Self {
        HEOptNodeType::Plain
    }
}

#[derive(Clone, Debug, Default)]
pub struct HEData {
    constval: Option<isize>,
    index_vars: HashSet<String>,
    node_type: HEOptNodeType,
    muldepth: usize,
}

impl Analysis<HEOptCircuit> for HEData {
    type Data = HEData;

    fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> DidMerge {
        let constval_merge = 
            merge_option(&mut to.constval, from.constval, |l, r| {
                if *l == r {
                    DidMerge(false, false)

                } else {
                    panic!("attempting to merge two eclasses with different constvals: {} and {}", l, r)
                }
            });

        let to_numvars = to.index_vars.len();
        to.index_vars.retain(|var| from.index_vars.contains(var));

        let vars_merge =
            DidMerge(
                to_numvars != to.index_vars.len(),
                to.index_vars != from.index_vars
            );

        let old_to_node_type = to.node_type;
        to.node_type = to.node_type.merge(&from.node_type);
        let type_merge =
            DidMerge(
                to.node_type != old_to_node_type,
                from.node_type != old_to_node_type
            );

        constval_merge | vars_merge | type_merge
    }

    fn make(egraph: &HEGraph, enode: &HEOptCircuit) -> Self::Data {
        let data = |id: &Id| &egraph[*id].data;

        match enode {
            HEOptCircuit::Literal(n) =>
                HEData {
                    constval: Some(*n),
                    index_vars: HashSet::new(),
                    node_type: HEOptNodeType::Plain,
                    muldepth: 0,
                },

            HEOptCircuit::Add([id1, id2]) => {
                let data1 = data(id1);
                let data2 = data(id2);

                let constval: Option<isize> =
                    data1.constval.and_then(|d1| {
                        data2.constval.map(|d2| d1 + d2)
                    });

                let mut index_vars: HashSet<String> = HashSet::new();
                index_vars.extend(data1.index_vars.clone());
                index_vars.extend(data2.index_vars.clone());

                HEData {
                    constval,
                    index_vars,
                    node_type: data1.node_type.ops(&data2.node_type),
                    muldepth: max(data1.muldepth, data2.muldepth),
                }
            },

            HEOptCircuit::Sub([id1, id2]) => {
                let data1 = data(id1);
                let data2 = data(id2);

                let constval: Option<isize> =
                    data1.constval.and_then(|d1| {
                        data2.constval.map(|d2| d1 - d2)
                    });

                let mut index_vars: HashSet<String> = HashSet::new();
                index_vars.extend(data1.index_vars.clone());
                index_vars.extend(data2.index_vars.clone());

                HEData {
                    constval,
                    index_vars,
                    node_type: data1.node_type.ops(&data2.node_type),
                    muldepth: max(data1.muldepth, data2.muldepth),
                }
            },

            HEOptCircuit::Mul([id1, id2]) => {
                let data1 = data(id1);
                let data2 = data(id2);

                let constval: Option<isize> =
                    data1.constval.and_then(|d1| {
                        data2.constval.map(|d2| d1 * d2)
                    });

                let mut index_vars: HashSet<String> = HashSet::new();
                index_vars.extend(data1.index_vars.clone());
                index_vars.extend(data2.index_vars.clone());

                let muldepth = 
                    match (data1.node_type, data2.node_type) {
                        (HEOptNodeType::Cipher, HEOptNodeType::Cipher) => 
                            max(data1.muldepth, data2.muldepth) + 1,

                        (HEOptNodeType::Cipher, HEOptNodeType::Plain) |
                        (HEOptNodeType::Plain, HEOptNodeType::Cipher) |
                        (HEOptNodeType::Plain, HEOptNodeType::Plain) => 
                            max(data1.muldepth, data2.muldepth) + 1,
                    };

                HEData {
                    constval,
                    index_vars,
                    node_type: data1.node_type.ops(&data2.node_type),
                    muldepth,
                }
            },

            HEOptCircuit::Rot([id1, id2]) => {
                let data1 = data(id1);
                let data2 = data(id2);

                HEData {
                    constval: data2.constval,
                    index_vars: data1.index_vars.clone(),
                    node_type: data2.node_type,
                    muldepth: data2.muldepth,
                }
            },

            HEOptCircuit::ProductVectors([id1, id2]) |
            HEOptCircuit::SumVectors([id1, id2]) => {
                let data1 = data(id1);
                let data2 = data(id2);
                let mut muldepth = max(data1.muldepth, data2.muldepth);

                if let HEOptCircuit::ProductVectors(_) = enode {
                    match (data1.node_type, data2.node_type) {
                        // this is supposed to be the extent of the reduced dim,
                        // but just approximate it with 1 for now
                        (HEOptNodeType::Cipher, HEOptNodeType::Cipher) =>  {
                            muldepth += 1;
                        },
                        _ => {}
                    }
                }

                HEData {
                    constval: None,
                    index_vars: data1.index_vars.clone(),
                    node_type: data2.node_type,
                    muldepth
                }
            },

            HEOptCircuit::IndexVar(var) => {
                HEData {
                    constval: None,
                    index_vars: HashSet::from([var.to_string()]),
                    node_type: HEOptNodeType::Plain,
                    muldepth: 0,
                }
            },

            HEOptCircuit::CiphertextVar(_) => {
                HEData {
                    constval: None,
                    index_vars: HashSet::new(),
                    node_type: HEOptNodeType::Cipher,
                    muldepth: 0,
                }
            },
            
            HEOptCircuit::PlaintextVar(_) |
            HEOptCircuit::FunctionVar(_, _) => {
                HEData {
                    constval: None,
                    index_vars: HashSet::new(),
                    node_type: HEOptNodeType::Plain,
                    muldepth: 0,
                }
            },
        }
    }

    fn modify(egraph: &mut EGraph<HEOptCircuit, Self>, id: Id) {
        // do constant folding
        if let Some(val) = egraph[id].data.constval {
            let val_id = egraph.add(HEOptCircuit::Literal(val));
            egraph.union(id, val_id);
        }
    }
}

fn is_not_index_var(var_str: &'static str) -> impl Fn(&mut HEGraph, Id, &Subst) -> bool {
    let var = var_str.parse().unwrap();
    move |egraph, _, subst| {
        egraph[subst[var]].nodes.iter().all(|node| {
            match node {
                HEOptCircuit::IndexVar(_) => false,
                _ => true,
            }
        })
    }
}

fn index_var_free(var1_str: &'static str, var2_str: &'static str) -> impl Fn(&mut HEGraph, Id, &Subst) -> bool {
    let var1: Var = var1_str.parse().unwrap();
    let var2: Var = var2_str.parse().unwrap();
    move |egraph, _, subst| {
        let index_vars1 = &egraph[subst[var1]].data.index_vars;
        let index_vars2= &egraph[subst[var2]].data.index_vars;
        index_vars2.iter().all(|ivar| !index_vars1.contains(ivar))
    }
}

// eclass has a constant value
fn is_const(var: &'static str) -> impl Fn(&mut HEGraph, Id, &Subst) -> bool {
    let var = var.parse().unwrap();
    move |egraph, _, subst| {
        egraph[subst[var]].data.constval.is_some()
    }
}

// eclass has a nonzero constant value
fn is_const_nonzero(var: &'static str) -> impl Fn(&mut HEGraph, Id, &Subst) -> bool {
    let var = var.parse().unwrap();
    move |egraph, _, subst| {
        match egraph[subst[var]].data.constval {
            Some(x) => x != 0,
            None => false,
        }
    }
}

struct ConstFold;

// do constant folding
impl Applier<HEOptCircuit, HEData> for ConstFold {
    fn apply_one(
        &self,
        egraph: &mut EGraph<HEOptCircuit, HEData>,
        eclass: Id,
        _subst: &Subst,
        _searcher_ast: Option<&PatternAst<HEOptCircuit>>,
        _rule_name: Symbol,
    ) -> Vec<Id> {
        let constval = egraph[eclass].data.constval.unwrap();
        let constval_id = egraph.add(HEOptCircuit::Literal(constval));
        if egraph.union(eclass, constval_id) {
            vec![eclass, constval_id]

        } else {
            vec![]
        }
    }
}

struct ConstSplit;

impl Applier<HEOptCircuit, HEData> for ConstSplit {
    fn apply_one(
        &self,
        egraph: &mut EGraph<HEOptCircuit, HEData>,
        eclass: Id,
        _subst: &Subst,
        _searcher_ast: Option<&PatternAst<HEOptCircuit>>,
        _rule_name: Symbol,
    ) -> Vec<Id> {
        let mut changed: Vec<Id> = Vec::new();
        let constval = egraph[eclass].data.constval.unwrap();

        if constval > 0 {
            let mut counter = 1;
            while counter <= constval {
                let op1_id = egraph.add(HEOptCircuit::Literal(counter));
                let op2_id = egraph.add(HEOptCircuit::Literal(constval - counter));
                let add_id = egraph.add(HEOptCircuit::Add([op1_id, op2_id]));
                if egraph.union(eclass, add_id) {
                    changed.push(add_id);
                }
                counter += 1;
            }

        } else if constval < 0 {
            let mut counter = -1;
            while counter >= constval {
                let op1_id = egraph.add(HEOptCircuit::Literal(counter));
                let op2_id = egraph.add(HEOptCircuit::Literal(constval - counter));
                let add_id = egraph.add(HEOptCircuit::Add([op1_id, op2_id]));
                if egraph.union(eclass, add_id) {
                    changed.push(add_id);
                }
                counter -= 1;
            }
        }

        if changed.len() > 0 {
            changed.push(eclass);
        }
        changed
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RotateWrap {
    size: usize,
    x: Var,
    l: Var,
}

impl Applier<HEOptCircuit, HEData> for RotateWrap {
    fn apply_one(
        &self,
        egraph: &mut HEGraph,
        matched_id: Id,
        subst: &Subst,
        _: Option<&PatternAst<HEOptCircuit>>,
        _: Symbol,
    ) -> Vec<Id> {
        let xclass: Id = subst[self.x];
        let lval = egraph[subst[self.l]].data.constval.unwrap();

        let wrapped_lval = egraph.add(HEOptCircuit::Literal(lval % (self.size as isize)));
        let wrapped_rot: Id = egraph.add(HEOptCircuit::Rot([xclass, wrapped_lval]));
        if egraph.union(matched_id, wrapped_rot) {
            vec![matched_id]

        } else {
            vec![]
        }
    }
}

pub struct Optimizer {
    rules: Vec<Rewrite<HEOptCircuit, HEData>>,
}

impl Optimizer {
    pub fn new(size: usize) -> Self {
        let mut rules: Vec<Rewrite<HEOptCircuit, HEData>> = vec![
            // bidirectional addition rules
            rewrite!("add-identity"; "(+ ?a 0)" <=> "?a" if is_not_index_var("?a")),
            rewrite!("add-assoc"; "(+ ?a (+ ?b ?c))" <=> "(+ (+ ?a ?b) ?c)"),
            rewrite!("add-commute"; "(+ ?a ?b)" <=> "(+ ?b ?a)"),
            rewrite!("add-to-two"; "(+ ?a ?a)" <=> "(* 2 ?a)"),
            rewrite!("sub-inverse"; "(- ?a ?b)" <=> "(+ ?a (* -1 ?b))"),

            // bidirectional multiplication rules
            rewrite!("mul-identity"; "(* ?a 1)" <=> "?a" if is_not_index_var("?a")),
            rewrite!("mul-assoc"; "(* ?a (* ?b ?c))" <=> "(* (* ?a ?b) ?c)"),
            rewrite!("mul-commute"; "(* ?a ?b)" <=> "(* ?b ?a)"),
            rewrite!("mul-distribute"; "(* (+ ?a ?b) ?c)" <=> "(+ (* ?a ?c) (* ?b ?c))"),

            // bidirectional rotation rules
            rewrite!("rot-distribute-mul"; "(rot ?o (* ?a ?b))" <=> "(* (rot ?o ?a) (rot ?o ?b))"),
            rewrite!("rot-distribute-add"; "(rot ?o (+ ?a ?b))" <=> "(+ (rot ?o ?a) (rot ?o ?b))"),
            rewrite!("rot-distribute-sub"; "(rot ?o (- ?a ?b))" <=> "(- (rot ?o ?a) (rot ?o ?b))"),

            rewrite!("sumvec-distribute-const-factor";
                "(sumvec ?o (* ?c ?x))" <=> "(* ?c (sumvec ?o ?x))"
                if is_const("?c")
            ),
        ]
        .concat();

        rules.extend(vec![
            // unidirectional rules
            rewrite!("mul-annihilator"; "(* ?a 0)" => "0"),
            rewrite!("sub-to-zero"; "(- ?a ?a)" => "0"),

            // constant folding rule
            rewrite!("constant-fold"; "?x" => ConstFold if is_const("?x")),

            // split a constant value into a sum 
            rewrite!("const-split"; "?x" => { ConstSplit {} } if is_const_nonzero("?x")),

            // rotation of 0 doesn't do anything
            rewrite!("rot-none"; "(rot 0 ?x)" => "?x"),

            // squash nested rotations into a single rotation
            rewrite!("rot-squash";
                "(rot ?o1 (rot ?o2 ?x))" => "(rot (+ ?o1 o2) ?x)"),

            // split rotations for complex offset expressions
            rewrite!("rot-offset-split-add";
                "(rot (+ ?o1 ?o2) ?x)" => "(rot ?o1 (rot ?o2 ?x))"),

            rewrite!("rot-offset-split-sub";
                "(rot (- ?o1 ?o2) ?x)" => "(rot ?o1 (rot (- 0 ?o2) ?x))"),

            // distribute rotations from reduction of vectors
            rewrite!("sumvec-distribute-rot";
                "(sumvec ?o1 (rot ?o2 ?x))" => "(rot ?o2 (sumvec ?o1 x))"
                if index_var_free("?o1", "?o2")
            ),

            rewrite!("prodvec-distribute-rot";
                "(prodvec ?o1 (rot ?o2 ?x))" => "(rot ?o2 (sumvec ?o1 x))"
                if index_var_free("?o1", "?o2")
            ),
        ]);

        Optimizer { rules }
    }

    pub fn rules(&self) -> &[Rewrite<HEOptCircuit, HEData>] {
        &self.rules
    }

    pub fn optimize(
        &self,
        exprs: Vec<RecExpr<HEOptCircuit>>,
        context: HECostContext,
        size: usize,
        timeout: usize,
        extractor_type: ExtractorType,
    ) -> Vec<RecExpr<HEOptCircuit>> {
        info!("running equality saturation for {} seconds...", timeout);

        let optimization_time = Instant::now();

        // simplify the expression using a Runner, which creates an e-graph with
        // the given expression and runs the given rules over it
        let mut runner =
            Runner::default()
            .with_node_limit(30000)
            .with_time_limit(Duration::from_secs(timeout as u64));

        for expr in exprs {
            runner = runner.with_expr(&expr);
        }

        runner = runner.run(&self.rules);

        info!("{}", runner.report().to_string());

        let roots = runner.roots.clone();
        let egraph = &mut runner.egraph;

        let extraction_time = Instant::now();

        let opt_exprs: Vec<RecExpr<HEOptCircuit>> = match extractor_type {
            ExtractorType::Greedy => {
                info!("using greedy extractor to derive optimized program...");
                // let extractor = GreedyExtractor::new(egraph, HECostFunction { egraph, count: 0 });
                // let extractor = Extractor::new(egraph, HECostFunction { egraph, latency: HELatencyModel::default() });
                let extractor = Extractor::new(
                    egraph,
                    HECostFunction {
                        latency: HELatencyModel::default(),
                        context,
                        egraph,
                    },
                );

                roots.into_iter().map(|root| {
                    let (_, opt_expr) = extractor.find_best(root);
                    opt_expr
                }).collect()
            },

            ExtractorType::LP => {
                info!("using LP extractor to derive optimized program...");
                // let mut lp_extractor = LpExtractor::new(egraph, OpSizeFunction { latency: HELatencyModel::default() });
                let mut lp_extractor =
                    LpExtractor::new(
                        egraph,
                        HELpCostFunction {
                            latency: HELatencyModel::default()
                        }
                    );
                let solution = lp_extractor.solve_multiple(&roots);
                // let mut lp_extractor = HEExtractor::new(egraph, root, HELatencyModel::default());
                // let solution = lp_extractor.solve();
                todo!()
            }
        };

        info!(
            "Extraction time: {}ms",
            extraction_time.elapsed().as_millis()
        );

        opt_exprs
    }
}

#[cfg(test)]
mod tests {
    use super::{*, dijkstra_extractor::DijkstraExtractor, cost::{HECostFunction, HELatencyModel}};
    use crate::{circ::{*, partial_eval::PlaintextHoisting, optimizer::cost::{HELpCostFunction, HECostContext}}, program::lowering::CircuitLowering};

    fn run_optimizer(expr: &RecExpr<HEOptCircuit>, duration_opt: Option<Duration>) -> Runner<HEOptCircuit, HEData> {
        let optimizer = Optimizer::new(16);
        let mut runner = Runner::default()
            .with_explanations_enabled()
            .with_expr(expr)
            .with_hook(|runner: &mut Runner<HEOptCircuit, HEData>| {
                println!("iteration: {}", runner.iterations.len());
                println!("nodes: {}", runner.egraph.total_number_of_nodes());
                Ok(())
            });
            

        let runner2 = 
            if let Some(duration) = duration_opt {
                runner.with_time_limit(duration)

            } else {
                runner
            };

        runner2.run(optimizer.rules())
    }

    fn run_equiv(s1: &str, s2: &str) -> (bool, String) {
        let expr1: RecExpr<HEOptCircuit> = s1.parse().unwrap();
        let expr2: RecExpr<HEOptCircuit> = s2.parse().unwrap();

        let mut runner = run_optimizer(&expr1, None);

        let equiv = runner.egraph.equivs(&expr1, &expr2).len() > 0;
        if equiv {
            (
                true,
                runner.explain_equivalence(&expr1, &expr2).get_flat_string(),
            )
        } else {
            (false, String::from(""))
        }
    }

    fn run_extractor(s: &str) {
        let optimizer = Optimizer::new(16);
        let expr = s.parse().unwrap();
        let runner = Runner::default()
            // .with_explanations_enabled()
            .with_expr(&expr)
            .with_hook(|runner: &mut Runner<HEOptCircuit, HEData>| {
                println!("iteration: {}", runner.iterations.len());
                println!("total nodes: {}", runner.egraph.total_number_of_nodes());
                Ok(())
            })
            .run(optimizer.rules());
        let root = *runner.roots.first().unwrap();

        let mut context =
            HECostContext {
                ct_multiplicity_map: HashMap::new(),
                pt_multiplicity_map: HashMap::new(),
                dim_extent_map: HashMap::new(),
            };

        for node in expr.as_ref() {
            match node {
                HEOptCircuit::CiphertextVar(var) => {
                    context.ct_multiplicity_map.insert(var.to_string(), 1);
                },

                HEOptCircuit::PlaintextVar(var) => {
                    context.pt_multiplicity_map.insert(var.to_string(), 1);
                },

                HEOptCircuit::SumVectors([ind_id, _]) |
                HEOptCircuit::ProductVectors([ind_id, _]) => {
                    let index =
                        runner.egraph[*ind_id].data.index_vars
                        .iter().next().unwrap().clone();

                    context.dim_extent_map.insert(index, 1);
                },

                _ => {}
            }
        }

        let cost_func =
            HECostFunction {
                latency: HELatencyModel::default(),
                context,
                egraph: &runner.egraph,
            };

        println!("extracting with greedy extractor");
        let extractor =
            DijkstraExtractor::new(&runner.egraph, cost_func);

        // let extractor = Extractor::new(&runner.egraph, cost_func);

        let (cost, solution) = extractor.find_best(root);

        println!("extracting with lp extractor");
        let mut lp_extractor =
            LpExtractor::new(
                &runner.egraph, 
                HELpCostFunction { latency: HELatencyModel::default() },
            );

        let lp_solution = lp_extractor.solve(root);
        println!("dijkstra extracted with cost {}: {}", cost, solution);
        println!("lp extracted: {}", lp_solution);
    }

    // #[ignore] ensures that these long-running equality saturation tests
    // don't run when calling `cargo test`.

    #[test]
    #[ignore]
    fn test_add_identity() {
        assert!(run_equiv("(+ x 0)", "x").0);
    }

    #[test]
    #[ignore]
    fn test_mul_to_add() {
        assert!(run_equiv("(* x 2)", "(+ x x)").0);
    }

    #[test]
    #[ignore]
    fn test_add_to_mul() {
        assert!(run_equiv("(+ (+ x x) x)", "(* x 3)").0);
    }

    #[test]
    #[ignore]
    fn test_factor() {
        assert!(run_equiv("(+ (* x x) (* 2 x))", "(* x (+ x 2))").0);
    }

    #[test]
    #[ignore]
    fn test_constant_fold() {
        assert!(run_equiv("(+ x (* 2 3))", "(+ x 6)").0);
    }

    #[test]
    #[ignore]
    fn test_neg_equiv() {
        assert!(!run_equiv("(+ (* x x) (* 2 x))", "(+ (+ x x) x)").0);
    }

    #[test]
    #[ignore]
    fn test_neg_equiv2() {
        let (equiv, explain) = run_equiv("(+ (* x x) (* 2 x))", "(* x (+ 3 x))");
        println!("{}", explain);
        assert!(!equiv);
    }

    #[test]
    #[ignore]
    fn test_extract() {
        run_extractor("(+ (* x x) (* 2 x))");
    }

    #[test]
    #[ignore]
    fn test_extract2() {
        run_extractor("(+ (+ 20 30) (+ (* x x) (* 2 x)))");
    }

    #[test]
    fn test_circuit_roundtrip() {
        let mut registry = CircuitObjectRegistry::new();

        let ct_var = registry.fresh_ct_var();
        let vector = VectorInfo {
            array: String::from("arr"),
            preprocessing: None,
            offset_map: BaseOffsetMap::new(2),
            dims: im::Vector::new(),
        };

        let ct_obj = CiphertextObject::InputVector(vector);

        let mut coord_map =
            IndexCoordinateMap::from_coord_system(IndexCoordinateSystem::from_dim_list(vec![
                (String::from("i"), 2),
                (String::from("j"), 2),
            ]));

        coord_map.set(im::vector![0, 0], ct_obj.clone());
        coord_map.set(im::vector![0, 1], ct_obj.clone());
        coord_map.set(im::vector![1, 0], ct_obj.clone());
        coord_map.set(im::vector![1, 1], ct_obj.clone());

        registry.set_ct_var_value(ct_var.clone(), CircuitValue::CoordMap(coord_map));

        let lit_2 = registry.register_circuit(ParamCircuitExpr::Literal(2));
        let add_2 = registry.register_circuit(ParamCircuitExpr::Op(Operator::Add, lit_2, lit_2));
        let ct = registry.register_circuit(ParamCircuitExpr::CiphertextVar(ct_var));
        let reduce_vec =
            registry.register_circuit(
                ParamCircuitExpr::ReduceDim(String::from("j"), 2, Operator::Add, ct)
            );

        let circuit =
            registry.register_circuit(ParamCircuitExpr::Op(Operator::Add, reduce_vec, add_2));

        let circuit_program = ParamCircuitProgram {
            registry,
            native_expr_list: vec![],
            circuit_expr_list: vec![(String::from("out"), vec![(String::from("i"), 2)], circuit)],
        };

        let (rec_exprs, _) = circuit_program.to_opt_circuit();
        let rec_expr = rec_exprs.first().unwrap();

        let runner =
            run_optimizer(&rec_expr, Some(Duration::from_millis(0)));

        let root = *runner.roots.first().unwrap();

        let get_first_enode = |id| {
            println!("fetching from eclass {}", id);
            runner.egraph[id].nodes[0].clone()
        };
        let new_expr =
            get_first_enode(root)
            .build_recexpr(get_first_enode);

        println!("{}", new_expr);

        let new_circuit_program = circuit_program.from_opt_circuit(vec![new_expr]);

        println!("old circuit:\n{}",
            CircuitLowering::new().run(
                PlaintextHoisting::new().run(circuit_program)
            )
        );
        println!("new circuit:\n{}", 
            CircuitLowering::new().run(
                PlaintextHoisting::new().run(new_circuit_program)
            )
        );
    }
}
