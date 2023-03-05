use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Display};
use priority_queue::PriorityQueue;

use egg::{Language, Analysis, Id, EGraph, RecExpr, CostFunction};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VisitedClasses(HashSet<Id>);

impl PartialOrd for VisitedClasses {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for VisitedClasses {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.len().cmp(&other.0.len())
    }
}

impl Display for VisitedClasses {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.len())
    }
}

/// use a version of Dijkstra's algorithm over hypergraphs for extraction
#[derive(Debug)]
pub struct DijkstraExtractor<'a, CF: CostFunction<L>, L: Language, N: Analysis<L>> {
    cost_function: CF,
    costs: HashMap<Id, (CF::Cost, L)>,
    egraph: &'a EGraph<L, N>,
}

impl<'a, CF, L, N> DijkstraExtractor<'a, CF, L, N>
where
    CF: CostFunction<L>,
    CF::Cost: Ord,
    L: Language,
    N: Analysis<L>,
{
    /// Create a new `Extractor` given an `EGraph` and a
    /// `CostFunction`.
    ///
    /// The extraction does all the work on creation, so this function
    /// performs the greedy search for cheapest representative of each
    /// eclass.
    pub fn new(egraph: &'a EGraph<L, N>, cost_function: CF) -> Self {
        let mut extractor = DijkstraExtractor {
            costs: HashMap::new(),
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

    fn get_node_cost(&mut self, eclass: Id, enode: &L) -> CF::Cost {
        self.cost_function.cost(
            enode,
            |id| self.costs[&self.egraph.find(id)].0.clone()
        )
    }

    fn find_costs(&mut self) {
        let mut class_tail_map: HashMap<Id, Vec<(Id, L)>> = HashMap::new();
        let mut node_counter_map: HashMap<(Id, L), usize> = HashMap::new();

        // populate class_tail_map and node_counter_map
        for class in self.egraph.classes() {
            for node in class.nodes.iter() {
                node_counter_map.insert((class.id, node.clone()), node.children().len());
                for child in node.children() {
                    let child_id = self.egraph.find(*child);

                    if !class_tail_map.contains_key(&child_id) {
                        class_tail_map.insert(child_id, vec![]);

                    } else {
                        class_tail_map.get_mut(&child_id).unwrap().push((class.id, node.clone()));
                    }
                }
            }
        }

        let mut visited: HashSet<Id> = HashSet::new();
        let mut queue: PriorityQueue<(Id, L), CF::Cost> = PriorityQueue::new();

        // add all enodes with no children
        for ((class, node), &cnt) in node_counter_map.iter() {
            if cnt == 0 {
                let cost = self.get_node_cost(*class, node);
                self.costs.insert(*class, (cost.clone(), node.clone()));
                queue.push((*class, node.clone()), cost);
            }
        }

        while !queue.is_empty() {
            // invariant: this is the first time that any enode in class has been
            // visited; thus the class has been visited, and we can decrement
            // the counter of all nodes with the class as its tail
            let ((class, node), cost) = queue.pop().unwrap();

            if !visited.contains(&class) {
                // add to visited
                visited.insert(class);

                // update cost
                self.costs.insert(class, (cost.clone(), node.clone()));

                // process reachable hyperedges
                for neighbor in class_tail_map[&class].iter() {
                    let cnt = node_counter_map.get_mut(neighbor).unwrap();
                    *cnt -= 1;

                    if *cnt == 0 {
                        let new_neighbor_cost = self.get_node_cost(neighbor.0, &neighbor.1);

                        // if neighbor is already in the queue,
                        // update priority if new cost is cheaper than old cost
                        if let Some((_, old_neighbor_cost)) = queue.get(neighbor) {
                            if new_neighbor_cost < *old_neighbor_cost {
                                queue.change_priority(neighbor, new_neighbor_cost);
                            }

                        } else {
                            queue.push(neighbor.clone(), new_neighbor_cost);
                        }
                    }
                }
            }
        }
    }
}