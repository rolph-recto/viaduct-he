use super::*;

/// object that takes an input schedule and returns a set of "nearby" schedules.
pub trait ScheduleTransformer {
    fn transform(&mut self, schedule: &Schedule) -> HashSet<Schedule>;
}
