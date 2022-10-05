use egg::*;
use std::{collections::HashMap, cmp::Ordering};
use crate::ir::{expr::HEExpr, optimizer::*};

#[derive(Debug, Clone)]
pub(crate) struct HECost {
    pub muldepth: usize,
    pub latency_map: HashMap<Id, usize>,
    pub cost: usize
}

impl HECost {
    fn calculate_cost(&mut self) -> usize {
        let mut total_latency: usize = 0;
        for v in self.latency_map.values() {
            total_latency += v;
        }

        self.cost = total_latency * self.muldepth;
        // self.cost = self.muldepth;
        self.cost
    }
}

impl PartialEq for HECost {
    fn eq(&self, other: &Self) -> bool {
        self.cost.eq(&other.cost)
    }
}

impl PartialOrd for HECost {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.cost.partial_cmp(&other.cost)
    }
}

pub(crate) struct HECostFunction<'a> {
    pub egraph: &'a HEGraph,
    pub count: usize
}

impl<'a> CostFunction<HEExpr> for HECostFunction<'a> {
    type Cost = HECost;

    fn cost<C>(&mut self, enode: &HEExpr, mut costs: C) -> Self::Cost
        where C: FnMut(Id) -> Self::Cost
    {
        let id = self.egraph.find(self.egraph.lookup(enode.clone()).unwrap());
        let self_data = self.egraph[id].data;
        
        let mut self_cost = HECost { muldepth: 0, latency_map: HashMap::new(), cost: 0 };
        for child in enode.children() {
            let child_id = self.egraph.find(*child);
            let child_cost = costs(child_id);

            self_cost.muldepth = max(self_cost.muldepth, child_cost.muldepth);

            for (k, v) in child_cost.latency_map.iter() {
                if !self_cost.latency_map.contains_key(k) {
                    self_cost.latency_map.insert(*k, *v);

                } else if self_cost.latency_map[k] > *v {
                    self_cost.latency_map.insert(*k, *v);
                }
            }
        }

        match *enode {
            HEExpr::Add(_) => {
                if self_data.constval.is_some() {
                    self_cost.latency_map.insert(id, ADD_PLAIN_LATENCY);

                } else {
                    self_cost.latency_map.insert(id, ADD_LATENCY);
                }
            },

            HEExpr::Mul(_) => {
                if self_data.constval.is_some() {
                    self_cost.latency_map.insert(id, MUL_PLAIN_LATENCY);

                } else {
                    self_cost.latency_map.insert(id, MUL_LATENCY);
                    self_cost.muldepth += 1;
                }
            },


            HEExpr::Num(_) => {
                self_cost.latency_map.insert(id, NUM_LATENCY);
            },

            HEExpr::Rot(_) => {
                self_cost.latency_map.insert(id, ROT_LATENCY);
            }

            HEExpr::Symbol(_) => {
                self_cost.latency_map.insert(id, SYM_LATENCY);
            }
        }

        self_cost.calculate_cost();
        self.count += 1;
        self_cost
    }
}


fn cmp<T: PartialOrd>(a: &Option<T>, b: &Option<T>) -> Ordering {
    // None is high
    match (a, b) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Greater,
        (Some(_), None) => Ordering::Less,
        (Some(a), Some(b)) => a.partial_cmp(b).unwrap(),
    }
}

#[derive(Debug)]
pub struct GreedyExtractor<'a, CF: CostFunction<L>, L: Language, N: Analysis<L>> {
    cost_function: CF,
    costs: HashMap<Id, (CF::Cost, L)>,
    egraph: &'a EGraph<L, N>,
}

impl<'a, CF, L, N> GreedyExtractor<'a, CF, L, N>
where
    CF: CostFunction<L>,
    L: Language,
    N: Analysis<L>,
{
    /// Create a new `ToposortExtractor` given an `EGraph` and a
    /// `CostFunction`.
    ///
    /// The extraction does all the work on creation, so this function
    /// performs the greedy search for cheapest representative of each
    /// eclass.
    pub fn new(egraph: &'a EGraph<L, N>, cost_function: CF) -> Self {
        let costs = HashMap::default();
        let mut extractor = GreedyExtractor {
            costs,
            egraph,
            cost_function,
        };
        extractor.find_costs();

        extractor
    }

    /// Find the cheapest (lowest cost) represented `RecExpr` in the
    /// given eclass.
    pub fn find_best(&self, eclass: Id) -> (CF::Cost, RecExpr<L>) {
        let (cost, root) = self.costs[&self.egraph.find(eclass)].clone();
        let expr = root.build_recexpr(|id| self.find_best_node(id).clone());
        (cost, expr)
    }

    /// Find the cheapest e-node in the given e-class.
    pub fn find_best_node(&self, eclass: Id) -> &L {
        &self.costs[&self.egraph.find(eclass)].1
    }

    /// Find the cost of the term that would be extracted from this e-class.
    pub fn find_best_cost(&self, eclass: Id) -> CF::Cost {
        let (cost, _) = &self.costs[&self.egraph.find(eclass)];
        cost.clone()
    }

    fn node_total_cost(&mut self, node: &L) -> Option<CF::Cost> {
        let eg = &self.egraph;
        let has_cost = |id| self.costs.contains_key(&eg.find(id));
        if node.all(has_cost) {
            let costs = &self.costs;
            let cost_f = |id| costs[&eg.find(id)].0.clone();
            Some(self.cost_function.cost(node, cost_f))
        } else {
            None
        }
    }

    fn find_costs(&mut self) {
        let mut did_something = true;
        while did_something {
            did_something = false;

            for class in self.egraph.classes() {
                let pass = self.make_pass(class);
                match (self.costs.get(&class.id), pass) {
                    (None, Some(new)) => {
                        self.costs.insert(class.id, new);
                        did_something = true;
                    }
                    (Some(old), Some(new)) if new.0 < old.0 => {
                        self.costs.insert(class.id, new);
                        did_something = true;
                    }
                    _ => (),
                }
            }
        }

        for class in self.egraph.classes() {
            if !self.costs.contains_key(&class.id) {
                log::warn!(
                    "Failed to compute cost for eclass {}: {:?}",
                    class.id,
                    class.nodes
                )
            }
        }
    }

    /// greedy algorithm to compute costs for each eclass;
    /// the classes are traversed in reverse toposort order so they are visited once
    /// (reverse toposort means that you count outdegrees instead of indegrees)
    /*
    fn find_costs(&mut self) {
        let mut outdegree_map: HashMap<Id, usize> = 
            HashMap::from_iter(
                self.egraph.classes().map(|cls| (cls.id, 0))
            );

        let mut parent_map: HashMap<Id, LinkedList<Id>> =
            HashMap::from_iter(
                self.egraph.classes().map(|cls| (cls.id, LinkedList::new()))
            );

        let mut id_map: HashMap<Id, &EClass<L, N::Data>> = HashMap::new();

        for cls in self.egraph.classes() {
            if cls.leaves().count() > 0 {
                id_map.insert(cls.id, cls);
                outdegree_map.insert(cls.id, 0);

            } else {
                let mut children: HashSet<Id> = HashSet::new();
                for node in cls.nodes.iter() {
                    for child_id in node.children().iter() {
                        let canon_child_id = self.egraph.find(*child_id);
                        if canon_child_id != cls.id {
                            children.insert(canon_child_id);
                        }
                    }
                }

                for child in children.iter() {
                    parent_map.get_mut(child).unwrap().push_back(cls.id);
                }

                id_map.insert(cls.id, cls);
                outdegree_map.insert(cls.id, children.len());
            }
        }

        while outdegree_map.len() > 0 {
            let mut cur_node_opt: Option<Id> = Option::None;
            for (k, v) in outdegree_map.iter() {
                if *v == 0 {
                    cur_node_opt = Some(*k);
                    break;
                }
            }

            let cur_node =
                cur_node_opt
                .unwrap_or_else(|| {
                    for k in outdegree_map.keys() {
                        println!("pass for {}, node: {:?}", k, id_map[k].nodes.first().unwrap());
                    }
                    panic!("cannot use toposort extractor on e-graph that has cycles");
                });

            println!("cur_node: {}", &cur_node);

            let cur_node_cost = self.make_pass(id_map[&cur_node]).unwrap();
            self.costs.insert(cur_node, cur_node_cost);

            let parents = &parent_map[&cur_node];
            for parent in parents {
                outdegree_map.insert(*parent, outdegree_map[parent] - 1);
            }

            outdegree_map.remove(&cur_node);
            println!("eclasses to process: {}", outdegree_map.len());
        }
    }
    */

    fn make_pass(&mut self, eclass: &EClass<L, N::Data>) -> Option<(CF::Cost, L)> {
        // explanation for why the filter below exists:
        // a single node is chosen as the representative of an eclass,
        // so we can't pick nodes with child edges to its own eclass,
        // otherwise egg will try to build an infinite expression.
        // e.g. x * 0 = 0, so if x * 0 is picked as the representative node,
        // egg will try to build the expression x * (x * (x * ...))!
        let (cost, node) = eclass
            .iter()
            .filter(|n| !n.children().contains(&eclass.id))
            .map(|n| (self.node_total_cost(n), n))
            .min_by(|a, b| cmp(&a.0, &b.0))
            .unwrap_or_else(|| panic!("Can't extract, eclass is empty: {:#?}", eclass));
        cost.map(|c| (c, node.clone()))
        /*
        if eclass.leaves().count() > 0 {
            let (cost, node) = 
                eclass
                .leaves()
                .map(|n| (self.node_total_cost(n), n))
                .min_by(|a, b| cmp(&a.0, &b.0))
                .unwrap_or_else(|| panic!("Can't extract, eclass is empty: {:#?}", eclass));
            cost.map(|c| (c, node.clone()))

        } else {
            let (cost, node) = eclass
                .iter()
                .map(|n| (self.node_total_cost(n), n))
                .min_by(|a, b| cmp(&a.0, &b.0))
                .unwrap_or_else(|| panic!("Can't extract, eclass is empty: {:#?}", eclass));
            cost.map(|c| (c, node.clone()))
        }
        */
    }
}