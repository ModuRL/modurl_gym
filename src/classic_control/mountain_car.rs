use candle_core::{Device, Tensor};
use modurl::{
    gym::{Gym, StepInfo},
    spaces::{self, Space},
};

/// The classic Mountain Car environment.
/// Converted from the OpenAI Gym Mountain Car environment.
pub struct MountainCarV0 {
    state: Tensor,
    action_space: spaces::Discrete,
    observation_space: spaces::BoxSpace,
    min_position: f32,
    max_position: f32,
    max_speed: f32,
    goal_position: f32,
    goal_velocity: f32,
    force: f32,
    gravity: f32,
}

impl MountainCarV0 {
    pub fn new(device: &Device) -> Self {
        let min_position = -1.2;
        let max_position = 0.6;
        let max_speed = 0.07;
        let goal_position = 0.5;
        let goal_velocity = 0.0;
        let force = 0.001;
        let gravity = 0.0025;

        let low = vec![min_position, -max_speed];
        let high = vec![max_position, max_speed];
        let low = Tensor::from_vec(low, vec![2], device).expect("Failed to create tensor.");
        let high = Tensor::from_vec(high, vec![2], device).expect("Failed to create tensor.");

        let action_space = spaces::Discrete::new(3, 0);
        let observation_space = spaces::BoxSpace::new(low, high);

        Self {
            state: Tensor::zeros(vec![2], candle_core::DType::F32, device)
                .expect("Failed to create tensor."),
            action_space,
            observation_space,
            min_position,
            max_position,
            max_speed,
            goal_position,
            goal_velocity,
            force,
            gravity,
        }
    }
}

impl Default for MountainCarV0 {
    fn default() -> Self {
        Self::new(&Device::Cpu)
    }
}

impl Gym for MountainCarV0 {
    type Error = candle_core::Error;

    fn get_name(&self) -> &str {
        "MountainCarV0"
    }

    fn reset(&mut self) -> Result<Tensor, Self::Error> {
        // Initialize position uniformly between -0.6 and -0.4
        let position = Tensor::rand(-0.6, -0.4, vec![1], self.state.device())?
            .to_dtype(candle_core::DType::F32)?;
        let velocity = Tensor::zeros(vec![1], candle_core::DType::F32, self.state.device())?;

        self.state = Tensor::cat(&[position, velocity], 0)?;
        Ok(self.state.clone())
    }

    fn step(&mut self, action: Tensor) -> Result<StepInfo, Self::Error> {
        assert!(self.action_space.contains(&action));

        let state_vec = self.state.to_vec1::<f32>()?;
        let (mut position, mut velocity) = (state_vec[0], state_vec[1]);

        let action_vec = action.to_vec0::<u32>()?;

        velocity +=
            (action_vec as f32 - 1.0) * self.force + (3.0 * position).cos() * (-self.gravity);

        velocity = velocity.clamp(-self.max_speed, self.max_speed);

        position += velocity;

        position = position.clamp(self.min_position, self.max_position);

        // Handle collision with left wall
        if position == self.min_position && velocity < 0.0 {
            velocity = 0.0;
        }

        self.state = Tensor::from_vec(vec![position, velocity], vec![2], self.state.device())?;

        // Check if goal is reached
        let terminated = position >= self.goal_position && velocity >= self.goal_velocity;
        let reward = -1.0;

        Ok(StepInfo {
            state: self.state.clone(),
            reward,
            done: terminated,
            truncated: false,
        })
    }

    fn observation_space(&self) -> Box<dyn Space> {
        Box::new(self.observation_space.clone())
    }

    fn action_space(&self) -> Box<dyn Space> {
        Box::new(self.action_space.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{Testable, test_gym_against_python};
    use Gym;

    #[test]
    fn test_mountain_car() {
        let mut env = MountainCarV0::new(&Device::Cpu);
        let state = env.reset().expect("Failed to reset environment.");
        assert_eq!(state.shape().dim(0).expect("Failed to get state dim."), 2);
        let StepInfo {
            state: next_state,
            reward,
            done,
            truncated: _truncated,
        } = env
            .step(
                Tensor::from_vec(vec![0 as u32], vec![], &Device::Cpu)
                    .expect("Failed to create tensor."),
            )
            .expect("Failed to step environment.");
        assert_eq!(
            next_state
                .shape()
                .dim(0)
                .expect("Failed to get next state dim."),
            2
        );
        assert!(reward == -1.0);
        assert!(!done);
    }

    #[test]
    #[should_panic]
    fn test_mountain_car_invalid_action() {
        let mut env = MountainCarV0::new(&Device::Cpu);
        let _state = env.reset();
        let _info = env
            .step(
                Tensor::from_vec(vec![3 as u32], vec![], &Device::Cpu)
                    .expect("Failed to create tensor."),
            )
            .expect("Failed to step environment.");
    }

    #[test]
    fn reward_is_negative_one_when_not_terminated() {
        let mut env = MountainCarV0::new(&Device::Cpu);
        env.reset().unwrap();
        let action = Tensor::from_vec(vec![1u32], vec![], &Device::Cpu).unwrap();
        let StepInfo {
            state: _state,
            reward,
            done,
            truncated: _truncated,
        } = env.step(action).unwrap();
        assert_eq!(reward, -1.0);
        assert!(!done);
    }

    impl Testable for MountainCarV0 {
        fn reset_deterministic(&mut self) -> Result<Tensor, candle_core::Error> {
            // Set deterministic initial state for testing (matches Python test with options={"low": 0.0, "high": 0.0})
            self.state = Tensor::from_vec(vec![0.0f32, 0.0], vec![2], &Device::Cpu)
                .expect("Failed to create tensor.");
            Ok(self.state.clone())
        }

        fn set_state(&mut self, state: Tensor, _: Option<serde_json::Value>) {
            self.state = state;
        }
    }

    #[test]
    fn test_mountain_car_against_python() {
        test_gym_against_python("mountain_car", MountainCarV0::new(&Device::Cpu), None);
    }
}
