use crate::circ::optimizer::*;
use coin_cbc::{Col, Model, Sense};
use good_lp::{solvers::coin_cbc::CoinCbcProblem, *};
use std::collections::{HashMap, HashSet};

use super::cost::HELatencyModel;
