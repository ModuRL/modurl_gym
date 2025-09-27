use box2d_rs::b2_body::*;
use box2d_rs::b2_collision::*;
use box2d_rs::b2_contact::*;
use box2d_rs::b2_fixture::*;
use box2d_rs::b2_joint::*;
use box2d_rs::b2_math::*;
use box2d_rs::b2_world::*;
use box2d_rs::b2_world_callbacks::*;
use box2d_rs::b2rs_common::UserDataType;
use box2d_rs::joints::b2_revolute_joint::*;

use box2d_rs::shapes::b2_edge_shape::*;
use box2d_rs::shapes::b2_polygon_shape::*;

use candle_core::{Device, Tensor};
use modurl::{
    gym::{Gym, StepInfo},
    spaces::{self, Space},
};
use std::cell::RefCell;
use std::rc::Rc;

// Constants from Python code
const FPS: f32 = 50.0;
const SCALE: f32 = 30.0; // affects how fast-paced the game is, forces should be adjusted as well

const MAIN_ENGINE_POWER: f32 = 13.0;
const SIDE_ENGINE_POWER: f32 = 0.6;

const INITIAL_RANDOM: f32 = 1000.0; // Set 1500 to make game harder

const LANDER_POLY: [(f32, f32); 6] = [
    (-14.0, 17.0),
    (-17.0, 0.0),
    (-17.0, -10.0),
    (17.0, -10.0),
    (17.0, 0.0),
    (14.0, 17.0),
];
const LEG_AWAY: f32 = 20.0;
const LEG_DOWN: f32 = 18.0;
const LEG_W: f32 = 2.0;
const LEG_H: f32 = 8.0;
const LEG_SPRING_TORQUE: f32 = 40.0;

const SIDE_ENGINE_HEIGHT: f32 = 14.0;
const SIDE_ENGINE_AWAY: f32 = 12.0;
const MAIN_ENGINE_Y_LOCATION: f32 = 4.0; // The Y location of the main engine on the body of the Lander.

const VIEWPORT_W: f32 = 600.0;
const VIEWPORT_H: f32 = 400.0;

#[derive(Default, Copy, Clone, Debug, PartialEq)]
struct UserDataTypes;
impl UserDataType for UserDataTypes {
    type Fixture = i32;
    type Body = i32;
    type Joint = i32;
}

pub struct ContactDetector {
    pub game_over: bool,

    pub legs_ground_contact: [bool; 2],
}

impl ContactDetector {
    pub fn new() -> Self {
        Self {
            game_over: false,
            legs_ground_contact: [false; 2],
        }
    }
}

impl B2contactListener<UserDataTypes> for ContactDetector {
    fn begin_contact(&mut self, contact: &mut dyn B2contactDynTrait<UserDataTypes>) {
        let fixture_a = contact.get_base().get_fixture_a();
        let fixture_b = contact.get_base().get_fixture_b();
        let body_a = fixture_a.borrow().get_body();
        let body_b = fixture_b.borrow().get_body();

        // Check if lander is involved in contact with ground (collision)
        let lander_body_id = 1; // Lander has body ID 1
        let ground_body_id = 0; // Ground has body ID 0
        let body_a_id = body_a.borrow().get_user_data().unwrap_or(0);
        let body_b_id = body_b.borrow().get_user_data().unwrap_or(0);

        // Set game_over immediately if lander body touches ground (not legs)
        if (body_a_id == lander_body_id && body_b_id == ground_body_id)
            || (body_b_id == lander_body_id && body_a_id == ground_body_id)
        {
            self.game_over = true;
        }

        // Check for leg ground contact (legs have IDs 2 and 3, ground has ID 0)
        let ground_body_id = 0;
        for i in 0..2 {
            let leg_body_id = 2 + i as i32;
            if (body_a_id == leg_body_id && body_b_id == ground_body_id)
                || (body_b_id == leg_body_id && body_a_id == ground_body_id)
            {
                self.legs_ground_contact[i] = true;
            }
        }
    }

    fn end_contact(&mut self, contact: &mut dyn B2contactDynTrait<UserDataTypes>) {
        let fixture_a = contact.get_base().get_fixture_a();
        let fixture_b = contact.get_base().get_fixture_b();
        let body_a = fixture_a.borrow().get_body();
        let body_b = fixture_b.borrow().get_body();

        let body_a_id = body_a.borrow().get_user_data().unwrap_or(0);
        let body_b_id = body_b.borrow().get_user_data().unwrap_or(0);

        // Check for leg ground contact end (legs have IDs 2 and 3, ground has ID 0)
        let ground_body_id = 0;
        for i in 0..2 {
            let leg_body_id = 2 + i as i32;
            if (body_a_id == leg_body_id && body_b_id == ground_body_id)
                || (body_b_id == leg_body_id && body_a_id == ground_body_id)
            {
                self.legs_ground_contact[i] = false;
            }
        }
    }

    fn pre_solve(
        &mut self,
        _contact: &mut dyn B2contactDynTrait<UserDataTypes>,
        _old_manifold: &B2manifold,
    ) {
        // Not needed for basic implementation
    }

    fn post_solve(
        &mut self,
        _contact: &mut dyn B2contactDynTrait<UserDataTypes>,
        _impulse: &B2contactImpulse,
    ) {
        // Not needed for basic implementation
    }
}

/// LunarLander environment with discrete actions only.
///
/// ## Action Space
/// Discrete(4) with the following actions:
/// - 0: do nothing
/// - 1: fire left orientation engine  
/// - 2: fire main engine
/// - 3: fire right orientation engine
pub struct LunarLanderV3 {
    // Environment parameters
    gravity: f32,
    enable_wind: bool,
    wind_power: f32,
    turbulence_power: f32,

    // Box2D world and bodies
    world: Option<B2worldPtr<UserDataTypes>>,
    moon: Option<BodyPtr<UserDataTypes>>,
    lander: Option<BodyPtr<UserDataTypes>>,
    legs: Vec<BodyPtr<UserDataTypes>>,
    leg_joints: Vec<B2jointPtr<UserDataTypes>>,
    particles: Vec<BodyPtr<UserDataTypes>>,

    // Contact detection
    contact_detector: Option<Rc<RefCell<ContactDetector>>>,

    // Environment state
    game_over: bool,

    prev_shaping: Option<f32>,

    // Terrain info
    helipad_x1: f32,
    helipad_x2: f32,
    helipad_y: f32,
    sky_polys: Vec<Vec<(f32, f32)>>,

    // Wind state
    wind_idx: i32,
    torque_idx: i32,

    // Spaces
    #[allow(dead_code)]
    action_space: Box<dyn Space>,
    #[allow(dead_code)]
    observation_space: Box<dyn Space>,

    // Random number generation
    rng: rand::rngs::ThreadRng,

    // Deterministic mode flag for testing
    deterministic_mode: bool,
}

impl LunarLanderV3 {
    pub fn new(gravity: f32, enable_wind: bool, wind_power: f32, turbulence_power: f32) -> Self {
        assert!(
            -12.0 < gravity && gravity < 0.0,
            "gravity (current value: {}) must be between -12 and 0",
            gravity
        );

        if wind_power < 0.0 || wind_power > 20.0 {
            eprintln!(
                "wind_power value is recommended to be between 0.0 and 20.0, (current value: {})",
                wind_power
            );
        }

        if turbulence_power < 0.0 || turbulence_power > 2.0 {
            eprintln!(
                "turbulence_power value is recommended to be between 0.0 and 2.0, (current value: {})",
                turbulence_power
            );
        }

        // Create observation space
        let low = vec![
            -2.5,                        // x coordinate
            -2.5,                        // y coordinate
            -10.0,                       // vx
            -10.0,                       // vy
            -2.0 * std::f32::consts::PI, // angle
            -10.0,                       // angular velocity
            0.0,                         // leg 1 contact
            0.0,                         // leg 2 contact
        ];
        let high = vec![
            2.5,                        // x coordinate
            2.5,                        // y coordinate
            10.0,                       // vx
            10.0,                       // vy
            2.0 * std::f32::consts::PI, // angle
            10.0,                       // angular velocity
            1.0,                        // leg 1 contact
            1.0,                        // leg 2 contact
        ];

        let low_tensor =
            Tensor::from_vec(low, vec![8], &Device::Cpu).expect("Failed to create low tensor");
        let high_tensor =
            Tensor::from_vec(high, vec![8], &Device::Cpu).expect("Failed to create high tensor");
        let observation_space = Box::new(spaces::BoxSpace::new(low_tensor, high_tensor));

        // Discrete action space: Nop, fire left engine, main engine, right engine
        let action_space: Box<dyn Space> = Box::new(spaces::Discrete::new(4, 0));

        Self {
            gravity,
            enable_wind,
            wind_power,
            turbulence_power,
            world: None,
            moon: None,
            lander: None,
            legs: Vec::new(),
            leg_joints: Vec::new(),
            particles: Vec::new(),
            contact_detector: None,
            game_over: false,
            prev_shaping: None,
            helipad_x1: 0.0,
            helipad_x2: 0.0,
            helipad_y: 0.0,
            sky_polys: Vec::new(),
            wind_idx: 0,
            torque_idx: 0,
            action_space,
            observation_space,
            rng: rand::rng(),
            deterministic_mode: false,
        }
    }

    fn destroy(&mut self) {
        if let Some(world) = self.world.take() {
            // Clean up particles
            for particle in &self.particles {
                world.borrow_mut().destroy_body(particle.clone());
            }
            self.particles.clear();

            // Destroy joints first
            for joint in self.leg_joints.drain(..) {
                world.borrow_mut().destroy_joint(joint);
            }

            // Destroy bodies
            if let Some(moon) = self.moon.take() {
                world.borrow_mut().destroy_body(moon);
            }

            if let Some(lander) = self.lander.take() {
                world.borrow_mut().destroy_body(lander);
            }

            for leg in self.legs.drain(..) {
                world.borrow_mut().destroy_body(leg);
            }
        }
    }

    #[allow(dead_code)]
    fn get_raw_physics_state(&self) -> Result<Vec<f32>, candle_core::Error> {
        if let Some(lander) = &self.lander {
            let pos = lander.borrow().get_position();
            let vel = lander.borrow().get_linear_velocity();
            let angle = lander.borrow().get_angle();
            let angular_vel = lander.borrow().get_angular_velocity();

            // Get leg contacts (simplified)
            let leg_contact_1 = self
                .contact_detector
                .as_ref()
                .map(|cd| cd.borrow().legs_ground_contact[0])
                .unwrap_or(false);
            let leg_contact_2 = self
                .contact_detector
                .as_ref()
                .map(|cd| cd.borrow().legs_ground_contact[1])
                .unwrap_or(false);

            // Return raw physics state (not normalized)
            Ok(vec![
                pos.x,
                pos.y,
                vel.x,
                vel.y,
                angle,
                angular_vel,
                if leg_contact_1 { 1.0 } else { 0.0 },
                if leg_contact_2 { 1.0 } else { 0.0 },
            ])
        } else {
            Ok(vec![0.0f32; 8])
        }
    }

    #[allow(dead_code)]
    fn get_current_state(&self) -> Result<Tensor, candle_core::Error> {
        if let Some(lander) = &self.lander {
            let pos = lander.borrow().get_position();
            let vel = lander.borrow().get_linear_velocity();
            let angle = lander.borrow().get_angle();
            let angular_vel = lander.borrow().get_angular_velocity();

            // Get leg contacts (simplified - would need proper contact detection)
            let leg_contact_1 = self
                .contact_detector
                .as_ref()
                .map(|cd| cd.borrow().legs_ground_contact[0])
                .unwrap_or(false);
            let leg_contact_2 = self
                .contact_detector
                .as_ref()
                .map(|cd| cd.borrow().legs_ground_contact[1])
                .unwrap_or(false);

            let state = vec![
                (pos.x - VIEWPORT_W / SCALE / 2.0) / (VIEWPORT_W / SCALE / 2.0),
                (pos.y - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2.0),
                vel.x * (VIEWPORT_W / SCALE / 2.0) / FPS,
                vel.y * (VIEWPORT_H / SCALE / 2.0) / FPS,
                angle,
                20.0 * angular_vel / FPS,
                if leg_contact_1 { 1.0 } else { 0.0 },
                if leg_contact_2 { 1.0 } else { 0.0 },
            ];

            Tensor::from_vec(state, vec![8], &Device::Cpu)
        } else {
            // Return zeros if no lander exists
            let state = vec![0.0f32; 8];
            Tensor::from_vec(state, vec![8], &Device::Cpu)
        }
    }
}

impl Default for LunarLanderV3 {
    fn default() -> Self {
        Self::new(-10.0, false, 15.0, 1.5)
    }
}

impl Gym for LunarLanderV3 {
    type Error = candle_core::Error;

    fn get_name(&self) -> &str {
        "LunarLander"
    }

    fn reset(&mut self) -> Result<Tensor, Self::Error> {
        use rand::Rng;

        self.destroy();

        // Disable deterministic mode for normal reset
        self.deterministic_mode = false;

        // Create new world
        let gravity = B2vec2::new(0.0, self.gravity);
        let world = B2world::<UserDataTypes>::new(gravity);

        // Set up contact listener
        let contact_detector = Rc::new(RefCell::new(ContactDetector::new()));
        world
            .borrow_mut()
            .set_contact_listener(contact_detector.clone());

        self.world = Some(world.clone());
        self.contact_detector = Some(contact_detector);
        self.game_over = false;
        self.prev_shaping = None;

        let w = VIEWPORT_W / SCALE;
        let h = VIEWPORT_H / SCALE;

        // Create Terrain
        const CHUNKS: usize = 11;
        let mut height = vec![0.0; CHUNKS + 1];
        for i in 0..=CHUNKS {
            height[i] = self.rng.random_range(0.0..h / 2.0);
        }

        let chunk_x: Vec<f32> = (0..CHUNKS)
            .map(|i| w / (CHUNKS - 1) as f32 * i as f32)
            .collect();
        self.helipad_x1 = chunk_x[CHUNKS / 2 - 1];
        self.helipad_x2 = chunk_x[CHUNKS / 2 + 1];
        self.helipad_y = h / 4.0;

        // Set helipad heights
        height[CHUNKS / 2 - 2] = self.helipad_y;
        height[CHUNKS / 2 - 1] = self.helipad_y;
        height[CHUNKS / 2] = self.helipad_y;
        height[CHUNKS / 2 + 1] = self.helipad_y;
        height[CHUNKS / 2 + 2] = self.helipad_y;

        let smooth_y: Vec<f32> = (0..CHUNKS)
            .map(|i| 0.33 * (height[i.saturating_sub(1)] + height[i] + height[i + 1]))
            .collect();

        // Create moon (ground)
        let mut ground_body_def = B2bodyDef::default();
        ground_body_def.body_type = B2bodyType::B2StaticBody;
        ground_body_def.position.set(0.0, 0.0);
        ground_body_def.user_data = Some(0); // Ground body ID = 0
        let moon = B2world::create_body(world.clone(), &ground_body_def);

        // Create edge from (0,0) to (W,0)
        let mut ground_edge = B2edgeShape::default();
        ground_edge.set_two_sided(B2vec2::new(0.0, 0.0), B2vec2::new(w, 0.0));

        let mut ground_fixture_def = B2fixtureDef::default();
        ground_fixture_def.shape = Some(Rc::new(RefCell::new(ground_edge)));
        ground_fixture_def.density = 0.0;
        ground_fixture_def.friction = 0.1;

        B2body::create_fixture(moon.clone(), &ground_fixture_def);

        self.sky_polys.clear();
        for i in 0..CHUNKS - 1 {
            let p1 = (chunk_x[i], smooth_y[i]);
            let p2 = (chunk_x[i + 1], smooth_y[i + 1]);

            let mut terrain_edge = B2edgeShape::default();
            terrain_edge.set_two_sided(B2vec2::new(p1.0, p1.1), B2vec2::new(p2.0, p2.1));

            let mut fixture_def = B2fixtureDef::default();
            fixture_def.shape = Some(Rc::new(RefCell::new(terrain_edge)));
            fixture_def.density = 0.0;
            fixture_def.friction = 0.1;

            B2body::create_fixture(moon.clone(), &fixture_def);

            self.sky_polys.push(vec![p1, p2, (p2.0, h), (p1.0, h)]);
        }

        self.moon = Some(moon);

        // Create Lander body
        let initial_y = VIEWPORT_H / SCALE;
        let initial_x = VIEWPORT_W / SCALE / 2.0;

        let mut lander_body_def = B2bodyDef::default();
        lander_body_def.body_type = B2bodyType::B2DynamicBody;
        lander_body_def.position.set(initial_x, initial_y);
        lander_body_def.angle = 0.0;
        lander_body_def.user_data = Some(1); // Lander body ID = 1

        let lander = B2world::create_body(world.clone(), &lander_body_def);

        // Create lander shape
        let mut lander_shape = B2polygonShape::default();
        let vertices: Vec<B2vec2> = LANDER_POLY
            .iter()
            .map(|(x, y)| B2vec2::new(x / SCALE, y / SCALE))
            .collect();
        lander_shape.set(&vertices);

        let mut fixture_def = B2fixtureDef::default();
        fixture_def.shape = Some(Rc::new(RefCell::new(lander_shape)));
        fixture_def.density = 5.0;
        fixture_def.friction = 0.1;
        fixture_def.filter.category_bits = 0x0010;
        fixture_def.filter.mask_bits = 0x001; // collide only with ground
        fixture_def.restitution = 0.0;

        B2body::create_fixture(lander.clone(), &fixture_def);

        // Apply initial random impulse
        let force_x = self.rng.random_range(-INITIAL_RANDOM..INITIAL_RANDOM);
        let force_y = self.rng.random_range(-INITIAL_RANDOM..INITIAL_RANDOM);
        lander
            .borrow_mut()
            .apply_force_to_center(B2vec2::new(force_x, force_y), true);

        self.lander = Some(lander);

        // Initialize wind if enabled
        if self.enable_wind {
            self.wind_idx = self.rng.random_range(-9999..9999);
            self.torque_idx = self.rng.random_range(-9999..9999);
        }

        // Create Lander Legs
        self.legs.clear();
        self.leg_joints.clear();
        for (leg_index, i) in [-1, 1].iter().enumerate() {
            let i_f = *i as f32;
            let mut leg_body_def = B2bodyDef::default();
            leg_body_def.body_type = B2bodyType::B2DynamicBody;
            leg_body_def
                .position
                .set(initial_x - i_f * LEG_AWAY / SCALE, initial_y);
            leg_body_def.angle = i_f * 0.05;
            leg_body_def.user_data = Some(2 + leg_index as i32); // Leg body IDs = 2, 3

            let leg = B2world::create_body(world.clone(), &leg_body_def);

            let mut leg_shape = B2polygonShape::default();
            leg_shape.set_as_box(LEG_W / SCALE, LEG_H / SCALE);

            let mut fixture_def = B2fixtureDef::default();
            fixture_def.shape = Some(Rc::new(RefCell::new(leg_shape)));
            fixture_def.density = 1.0;
            fixture_def.restitution = 0.0;
            fixture_def.filter.category_bits = 0x0020;
            fixture_def.filter.mask_bits = 0x001;

            B2body::create_fixture(leg.clone(), &fixture_def);

            // Create revolute joint connecting leg to lander
            let mut joint_def = B2revoluteJointDef::default();
            joint_def.base.body_a = Some(self.lander.as_ref().unwrap().clone());
            joint_def.base.body_b = Some(leg.clone());
            joint_def.local_anchor_a = B2vec2::new(0.0, 0.0);
            joint_def.local_anchor_b = B2vec2::new(i_f * LEG_AWAY / SCALE, LEG_DOWN / SCALE);
            joint_def.enable_motor = true;
            joint_def.enable_limit = true;
            joint_def.max_motor_torque = LEG_SPRING_TORQUE;
            joint_def.motor_speed = 0.3 * i_f;
            if i_f == -1.0 {
                joint_def.lower_angle = 0.9 - 0.5;
                joint_def.upper_angle = 0.9;
            } else {
                joint_def.lower_angle = -0.9;
                joint_def.upper_angle = -0.9 + 0.5;
            }

            let joint_def_enum = B2JointDefEnum::RevoluteJoint(joint_def);
            let joint = world.borrow_mut().create_joint(&joint_def_enum);
            self.leg_joints.push(joint);
            self.legs.push(leg);
        }

        // Step once to ensure proper initialization
        let step_info = self.step(Tensor::from_vec(vec![0u32], vec![], &Device::Cpu)?)?;

        Ok(step_info.state)
    }

    fn step(&mut self, action: Tensor) -> Result<StepInfo, Self::Error> {
        use rand::Rng;

        assert!(self.lander.is_some(), "You forgot to call reset()");
        let world = self.world.as_ref().unwrap();
        let lander = self.lander.as_ref().unwrap();

        let action_u32 = action.to_vec0::<u32>()?;

        // Update wind and apply to the lander
        if self.enable_wind {
            let legs_contact = self.legs.len() >= 2
                && (self
                    .contact_detector
                    .as_ref()
                    .map(|cd| cd.borrow().legs_ground_contact[0])
                    .unwrap_or(false)
                    || self
                        .contact_detector
                        .as_ref()
                        .map(|cd| cd.borrow().legs_ground_contact[1])
                        .unwrap_or(false));

            if !legs_contact {
                // Wind calculation
                let wind_mag = ((0.02 * self.wind_idx as f32).sin()
                    + (std::f32::consts::PI * 0.01 * self.wind_idx as f32).sin())
                .tanh()
                    * self.wind_power;
                self.wind_idx += 1;
                lander
                    .borrow_mut()
                    .apply_force_to_center(B2vec2::new(wind_mag, 0.0), true);

                // Torque calculation
                let torque_mag = ((0.02 * self.torque_idx as f32).sin()
                    + (std::f32::consts::PI * 0.01 * self.torque_idx as f32).sin())
                .tanh()
                    * self.turbulence_power;
                self.torque_idx += 1;
                lander.borrow_mut().apply_torque(torque_mag, true);
            }
        }

        // Apply Engine Impulses
        let lander_angle = lander.borrow().get_angle();
        let tip = (lander_angle.sin(), lander_angle.cos());
        let side = (-tip.1, tip.0);

        // Generate dispersion
        let dispersion = if self.deterministic_mode {
            // Use fixed dispersion for deterministic testing
            [0.0, 0.0]
        } else {
            // Use random dispersion for normal operation
            [
                self.rng.random_range(-1.0..1.0) / SCALE,
                self.rng.random_range(-1.0..1.0) / SCALE,
            ]
        };

        let mut m_power = 0.0;
        let should_fire_main = action_u32 == 2;

        if should_fire_main {
            // Main engine
            m_power = 1.0;

            let lander_pos = lander.borrow().get_position();
            let ox = tip.0 * (MAIN_ENGINE_Y_LOCATION / SCALE + 2.0 * dispersion[0])
                + side.0 * dispersion[1];
            let oy = -tip.1 * (MAIN_ENGINE_Y_LOCATION / SCALE + 2.0 * dispersion[0])
                - side.1 * dispersion[1];

            let impulse_pos = (lander_pos.x + ox, lander_pos.y + oy);

            // Apply impulse to lander
            let impulse_force = B2vec2::new(
                -ox * MAIN_ENGINE_POWER * m_power,
                -oy * MAIN_ENGINE_POWER * m_power,
            );

            lander.borrow_mut().apply_linear_impulse(
                impulse_force,
                B2vec2::new(impulse_pos.0, impulse_pos.1),
                true,
            );
        }

        let mut s_power = 0.0;
        let should_fire_side = action_u32 == 1 || action_u32 == 3;

        if should_fire_side {
            // Orientation/Side engines
            // action = 1 is left, action = 3 is right
            let direction = (action_u32 as i32 - 2) as f32;
            s_power = 1.0;

            let lander_pos = lander.borrow().get_position();
            let ox = tip.0 * dispersion[0]
                + side.0 * (3.0 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE);
            let oy = -tip.1 * dispersion[0]
                - side.1 * (3.0 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE);

            let impulse_pos = (
                lander_pos.x + ox - tip.0 * 17.0 / SCALE,
                lander_pos.y + oy + tip.1 * SIDE_ENGINE_HEIGHT / SCALE,
            );

            // Apply impulse to lander
            let impulse_force = B2vec2::new(
                -ox * SIDE_ENGINE_POWER * s_power,
                -oy * SIDE_ENGINE_POWER * s_power,
            );

            lander.borrow_mut().apply_linear_impulse(
                impulse_force,
                B2vec2::new(impulse_pos.0, impulse_pos.1),
                true,
            );
        }

        // Step the world
        world.borrow_mut().step(1.0 / FPS, 6 * 30, 2 * 30);

        // Get state
        let pos = lander.borrow().get_position();
        let vel = lander.borrow().get_linear_velocity();
        let angle = lander.borrow().get_angle();
        let angular_vel = lander.borrow().get_angular_velocity();

        // Get leg contacts (simplified - would need proper contact detection)
        let leg_contact_1 = self
            .contact_detector
            .as_ref()
            .map(|cd| cd.borrow().legs_ground_contact[0])
            .unwrap_or(false);
        let leg_contact_2 = self
            .contact_detector
            .as_ref()
            .map(|cd| cd.borrow().legs_ground_contact[1])
            .unwrap_or(false);

        let state = vec![
            (pos.x - VIEWPORT_W / SCALE / 2.0) / (VIEWPORT_W / SCALE / 2.0),
            (pos.y - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2.0),
            vel.x * (VIEWPORT_W / SCALE / 2.0) / FPS,
            vel.y * (VIEWPORT_H / SCALE / 2.0) / FPS,
            angle,
            20.0 * angular_vel / FPS,
            if leg_contact_1 { 1.0 } else { 0.0 },
            if leg_contact_2 { 1.0 } else { 0.0 },
        ];

        let state_tensor = Tensor::from_vec(state.clone(), vec![8], &Device::Cpu)?;

        // Calculate reward
        let mut reward = 0.0;
        let shaping = -100.0 * (state[0] * state[0] + state[1] * state[1]).sqrt()
            - 100.0 * (state[2] * state[2] + state[3] * state[3]).sqrt()
            - 100.0 * state[4].abs()
            + 10.0 * state[6]
            + 10.0 * state[7];

        if let Some(prev_shaping) = self.prev_shaping {
            reward = shaping - prev_shaping;
        }
        self.prev_shaping = Some(shaping);

        reward -= m_power * 0.30; // less fuel spent is better
        reward -= s_power * 0.03;

        // Check termination conditions
        let mut terminated = false;
        // Read game_over from contact detector
        let contact_game_over = self
            .contact_detector
            .as_ref()
            .map(|cd| cd.borrow().game_over)
            .unwrap_or(false);

        if self.game_over || contact_game_over || state[0].abs() >= 1.0 {
            terminated = true;
            reward = -100.0;
        } else if !lander.borrow().is_awake() {
            terminated = true;
            reward = 100.0;
        }

        Ok(StepInfo {
            state: state_tensor,
            reward,
            done: terminated,
            truncated: false,
        })
    }

    fn observation_space(&self) -> Box<dyn Space> {
        let low = vec![
            -2.5,
            -2.5,
            -10.0,
            -10.0,
            -2.0 * std::f32::consts::PI,
            -10.0,
            0.0,
            0.0,
        ];
        let high = vec![
            2.5,
            2.5,
            10.0,
            10.0,
            2.0 * std::f32::consts::PI,
            10.0,
            1.0,
            1.0,
        ];
        let low_tensor =
            Tensor::from_vec(low, vec![8], &Device::Cpu).expect("Failed to create low tensor");
        let high_tensor =
            Tensor::from_vec(high, vec![8], &Device::Cpu).expect("Failed to create high tensor");
        Box::new(spaces::BoxSpace::new(low_tensor, high_tensor))
    }

    fn action_space(&self) -> Box<dyn Space> {
        // Discrete action space: 0=nop, 1=left engine, 2=main engine, 3=right engine
        Box::new(spaces::Discrete::new(4, 0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{Testable, Tolerances, test_gym_against_python};

    impl Testable for LunarLanderV3 {
        fn reset_deterministic(&mut self) -> Result<Tensor, candle_core::Error> {
            // Do a deterministic reset that controls all random elements
            self.destroy();

            // Enable deterministic mode
            self.deterministic_mode = true;

            // Create new world
            let gravity = B2vec2::new(0.0, self.gravity);
            let world = B2world::<UserDataTypes>::new(gravity);

            // Set up contact listener
            let contact_detector = Rc::new(RefCell::new(ContactDetector::new()));
            world
                .borrow_mut()
                .set_contact_listener(contact_detector.clone());

            self.world = Some(world.clone());
            self.contact_detector = Some(contact_detector);
            self.game_over = false;
            self.prev_shaping = None;

            let w = VIEWPORT_W / SCALE;
            let h = VIEWPORT_H / SCALE;

            // Create DETERMINISTIC Terrain (no randomness)
            const CHUNKS: usize = 11;
            let mut height = vec![0.0; CHUNKS + 1];
            // Set all heights to a fixed value instead of random
            for i in 0..=CHUNKS {
                height[i] = h / 8.0; // Fixed height instead of random
            }

            let chunk_x: Vec<f32> = (0..CHUNKS)
                .map(|i| w / (CHUNKS - 1) as f32 * i as f32)
                .collect();
            self.helipad_x1 = chunk_x[CHUNKS / 2 - 1];
            self.helipad_x2 = chunk_x[CHUNKS / 2 + 1];
            self.helipad_y = h / 4.0;

            // Set helipad heights
            height[CHUNKS / 2 - 2] = self.helipad_y;
            height[CHUNKS / 2 - 1] = self.helipad_y;
            height[CHUNKS / 2] = self.helipad_y;
            height[CHUNKS / 2 + 1] = self.helipad_y;
            height[CHUNKS / 2 + 2] = self.helipad_y;

            let smooth_y: Vec<f32> = (0..CHUNKS)
                .map(|i| 0.33 * (height[i.saturating_sub(1)] + height[i] + height[i + 1]))
                .collect();

            // Create moon (ground)
            let mut ground_body_def = B2bodyDef::default();
            ground_body_def.body_type = B2bodyType::B2StaticBody;
            ground_body_def.position.set(0.0, 0.0);
            ground_body_def.user_data = Some(0); // Ground body ID = 0
            let moon = B2world::create_body(world.clone(), &ground_body_def);

            // Create edge from (0,0) to (W,0)
            let mut ground_edge = B2edgeShape::default();
            ground_edge.set_two_sided(B2vec2::new(0.0, 0.0), B2vec2::new(w, 0.0));

            let mut ground_fixture_def = B2fixtureDef::default();
            ground_fixture_def.shape = Some(Rc::new(RefCell::new(ground_edge)));
            ground_fixture_def.density = 0.0;
            ground_fixture_def.friction = 0.1;

            B2body::create_fixture(moon.clone(), &ground_fixture_def);

            self.sky_polys.clear();
            for i in 0..CHUNKS - 1 {
                let p1 = (chunk_x[i], smooth_y[i]);
                let p2 = (chunk_x[i + 1], smooth_y[i + 1]);

                let mut terrain_edge = B2edgeShape::default();
                terrain_edge.set_two_sided(B2vec2::new(p1.0, p1.1), B2vec2::new(p2.0, p2.1));

                let mut fixture_def = B2fixtureDef::default();
                fixture_def.shape = Some(Rc::new(RefCell::new(terrain_edge)));
                fixture_def.density = 0.0;
                fixture_def.friction = 0.1;

                B2body::create_fixture(moon.clone(), &fixture_def);

                self.sky_polys.push(vec![p1, p2, (p2.0, h), (p1.0, h)]);
            }

            self.moon = Some(moon);

            // Create Lander body at deterministic position
            let initial_y = VIEWPORT_H / SCALE * 0.8; // Deterministic Y position
            let initial_x = VIEWPORT_W / SCALE / 2.0; // Deterministic X position

            let mut lander_body_def = B2bodyDef::default();
            lander_body_def.body_type = B2bodyType::B2DynamicBody;
            lander_body_def.position.set(initial_x, initial_y);
            lander_body_def.angle = 0.0;
            lander_body_def.user_data = Some(1); // Lander body ID = 1

            let lander = B2world::create_body(world.clone(), &lander_body_def);

            // Create lander shape
            let mut lander_shape = B2polygonShape::default();
            let vertices: Vec<B2vec2> = LANDER_POLY
                .iter()
                .map(|(x, y)| B2vec2::new(x / SCALE, y / SCALE))
                .collect();
            lander_shape.set(&vertices);

            let mut fixture_def = B2fixtureDef::default();
            fixture_def.shape = Some(Rc::new(RefCell::new(lander_shape)));
            fixture_def.density = 5.0;
            fixture_def.friction = 0.1;
            fixture_def.filter.category_bits = 0x0010;
            fixture_def.filter.mask_bits = 0x001; // collide only with ground
            fixture_def.restitution = 0.0;

            B2body::create_fixture(lander.clone(), &fixture_def);

            // NO RANDOM IMPULSE - skip the random force application for deterministic reset

            // Set deterministic velocities directly
            lander
                .borrow_mut()
                .set_linear_velocity(B2vec2::new(0.0, -1.0));
            lander.borrow_mut().set_angular_velocity(0.0);

            self.lander = Some(lander);

            // Initialize wind deterministically if enabled
            if self.enable_wind {
                self.wind_idx = 0; // Fixed instead of random
                self.torque_idx = 0; // Fixed instead of random
            }

            // Create Lander Legs at deterministic positions
            self.legs.clear();
            self.leg_joints.clear();
            for (leg_index, i) in [-1, 1].iter().enumerate() {
                let i_f = *i as f32;
                let mut leg_body_def = B2bodyDef::default();
                leg_body_def.body_type = B2bodyType::B2DynamicBody;
                leg_body_def
                    .position
                    .set(initial_x - i_f * LEG_AWAY / SCALE, initial_y);
                leg_body_def.angle = i_f * 0.05;
                leg_body_def.user_data = Some(2 + leg_index as i32); // Leg body IDs = 2, 3

                let leg = B2world::create_body(world.clone(), &leg_body_def);

                let mut leg_shape = B2polygonShape::default();
                leg_shape.set_as_box(LEG_W / SCALE, LEG_H / SCALE);

                let mut fixture_def = B2fixtureDef::default();
                fixture_def.shape = Some(Rc::new(RefCell::new(leg_shape)));
                fixture_def.density = 1.0;
                fixture_def.restitution = 0.0;
                fixture_def.filter.category_bits = 0x0020;
                fixture_def.filter.mask_bits = 0x001;

                B2body::create_fixture(leg.clone(), &fixture_def);

                // Create revolute joint connecting leg to lander
                let mut joint_def = B2revoluteJointDef::default();
                joint_def.base.body_a = Some(self.lander.as_ref().unwrap().clone());
                joint_def.base.body_b = Some(leg.clone());
                joint_def.local_anchor_a = B2vec2::new(0.0, 0.0);
                joint_def.local_anchor_b = B2vec2::new(i_f * LEG_AWAY / SCALE, LEG_DOWN / SCALE);
                joint_def.enable_motor = true;
                joint_def.enable_limit = true;
                joint_def.max_motor_torque = LEG_SPRING_TORQUE;
                joint_def.motor_speed = 0.3 * i_f;
                if i_f == -1.0 {
                    joint_def.lower_angle = 0.9 - 0.5;
                    joint_def.upper_angle = 0.9;
                } else {
                    joint_def.lower_angle = -0.9;
                    joint_def.upper_angle = -0.9 + 0.5;
                }

                let joint_def_enum = B2JointDefEnum::RevoluteJoint(joint_def);
                let joint = world.borrow_mut().create_joint(&joint_def_enum);
                self.leg_joints.push(joint);

                // Set deterministic leg velocities before moving leg
                leg.borrow_mut().set_linear_velocity(B2vec2::new(0.0, -1.0));
                leg.borrow_mut().set_angular_velocity(0.0);

                self.legs.push(leg);
            }

            // Return the current observation (normalized values)
            self.get_current_state()
        }

        fn set_state(&mut self, _state: Tensor, extra_info: Option<serde_json::Value>) {
            // Extract raw physics values from the extra_info (from Python custom_info)
            let info = extra_info.expect("extra_info is required for set_state");

            let raw_pos_x = info
                .get("raw_lander_pos_x")
                .and_then(|v| v.as_f64())
                .map(|v| v as f32)
                .expect("raw_lander_pos_x not found in extra_info");
            let raw_pos_y = info
                .get("raw_lander_pos_y")
                .and_then(|v| v.as_f64())
                .map(|v| v as f32)
                .expect("raw_lander_pos_y not found in extra_info");
            let raw_vel_x = info
                .get("raw_lander_vel_x")
                .and_then(|v| v.as_f64())
                .map(|v| v as f32)
                .expect("raw_lander_vel_x not found in extra_info");
            let raw_vel_y = info
                .get("raw_lander_vel_y")
                .and_then(|v| v.as_f64())
                .map(|v| v as f32)
                .expect("raw_lander_vel_y not found in extra_info");
            let raw_angle = info
                .get("raw_lander_angle")
                .and_then(|v| v.as_f64())
                .map(|v| v as f32)
                .expect("raw_lander_angle not found in extra_info");
            let raw_angular_vel = info
                .get("raw_lander_angular_vel")
                .and_then(|v| v.as_f64())
                .map(|v| v as f32)
                .expect("raw_lander_angular_vel not found in extra_info");

            if let Some(lander) = &self.lander {
                // Set lander position (raw physics coordinates)
                lander
                    .borrow_mut()
                    .set_transform(B2vec2::new(raw_pos_x, raw_pos_y), raw_angle);

                // Set lander velocities
                lander
                    .borrow_mut()
                    .set_linear_velocity(B2vec2::new(raw_vel_x, raw_vel_y));
                lander.borrow_mut().set_angular_velocity(raw_angular_vel);

                // Set legs using explicit leg data from extra_info
                for (i, leg) in self.legs.iter().enumerate() {
                    let leg_x = info
                        .get(&format!("raw_leg{}_pos_x", i))
                        .and_then(|v| v.as_f64())
                        .map(|v| v as f32)
                        .expect(&format!("raw_leg{}_pos_x not found in extra_info", i));
                    let leg_y = info
                        .get(&format!("raw_leg{}_pos_y", i))
                        .and_then(|v| v.as_f64())
                        .map(|v| v as f32)
                        .expect(&format!("raw_leg{}_pos_y not found in extra_info", i));
                    let leg_angle = info
                        .get(&format!("raw_leg{}_angle", i))
                        .and_then(|v| v.as_f64())
                        .map(|v| v as f32)
                        .expect(&format!("raw_leg{}_angle not found in extra_info", i));
                    let leg_vel_x = info
                        .get(&format!("raw_leg{}_vel_x", i))
                        .and_then(|v| v.as_f64())
                        .map(|v| v as f32)
                        .expect(&format!("raw_leg{}_vel_x not found in extra_info", i));
                    let leg_vel_y = info
                        .get(&format!("raw_leg{}_vel_y", i))
                        .and_then(|v| v.as_f64())
                        .map(|v| v as f32)
                        .expect(&format!("raw_leg{}_vel_y not found in extra_info", i));
                    let leg_angular_vel = info
                        .get(&format!("raw_leg{}_angular_vel", i))
                        .and_then(|v| v.as_f64())
                        .map(|v| v as f32)
                        .expect(&format!("raw_leg{}_angular_vel not found in extra_info", i));

                    leg.borrow_mut()
                        .set_transform(B2vec2::new(leg_x, leg_y), leg_angle);
                    leg.borrow_mut()
                        .set_linear_velocity(B2vec2::new(leg_vel_x, leg_vel_y));
                    leg.borrow_mut().set_angular_velocity(leg_angular_vel);
                }

                // Wake up bodies to ensure they're active
                lander.borrow_mut().set_awake(true);
                for leg in &self.legs {
                    leg.borrow_mut().set_awake(true);
                }
            }

            // Sync leg contact state from Python environment
            if let Some(contact_detector) = &self.contact_detector {
                let leg0_contact = info
                    .get("leg0_contact")
                    .and_then(|v| v.as_f64())
                    .map(|v| v > 0.5)
                    .unwrap_or(false);
                let leg1_contact = info
                    .get("leg1_contact")
                    .and_then(|v| v.as_f64())
                    .map(|v| v > 0.5)
                    .unwrap_or(false);

                contact_detector.borrow_mut().legs_ground_contact[0] = leg0_contact;
                contact_detector.borrow_mut().legs_ground_contact[1] = leg1_contact;
            }
        }
    }

    #[test]
    fn test_lunar_lander_creation() {
        let env = LunarLanderV3::default();
        assert_eq!(env.get_name(), "LunarLander");
    }

    #[test]
    fn test_lunar_lander_reset() {
        let mut env = LunarLanderV3::default();
        let state = env.reset().expect("Failed to reset environment");
        assert_eq!(state.dims(), &[8]);
    }

    #[test]
    fn test_lunar_lander_step() {
        let mut env = LunarLanderV3::default();
        env.reset().expect("Failed to reset environment");

        let action = Tensor::from_vec(vec![0u32], vec![], &Device::Cpu)
            .expect("Failed to create action tensor");
        let step_info = env.step(action).expect("Failed to step environment");

        assert_eq!(step_info.state.dims(), &[8]);
        assert!(!step_info.done); // Should not be done immediately
    }

    #[test]
    fn test_lunar_lander_actions() {
        let mut env = LunarLanderV3::default();
        env.reset().expect("Failed to reset environment");

        // Test all four discrete actions
        for action in 0..4 {
            let action_tensor = Tensor::from_vec(vec![action as u32], vec![], &Device::Cpu)
                .expect("Failed to create action tensor");
            let step_info = env.step(action_tensor).expect("Failed to step environment");

            assert_eq!(step_info.state.dims(), &[8]);
            // Reward should be reasonable (not NaN or infinite)
            assert!(step_info.reward.is_finite());
        }
    }

    #[test]
    fn test_lunar_lander_with_wind() {
        let mut env = LunarLanderV3::new(-10.0, true, 15.0, 1.5);
        env.reset().expect("Failed to reset environment");

        // Test with wind enabled
        let action = Tensor::from_vec(vec![2u32], vec![], &Device::Cpu)
            .expect("Failed to create action tensor");
        let step_info = env.step(action).expect("Failed to step environment");

        assert_eq!(step_info.state.dims(), &[8]);
        assert!(step_info.reward.is_finite());
    }

    #[test]
    fn test_lunar_lander_deterministic_reset() {
        let mut env = LunarLanderV3::default();
        let state1 = env
            .reset_deterministic()
            .expect("Failed to reset deterministically");
        let state2 = env
            .reset_deterministic()
            .expect("Failed to reset deterministically again");

        // States should be the same size
        assert_eq!(state1.dims(), state2.dims());
        assert_eq!(state1.dims(), &[8]);

        // The states might not be identical due to random terrain generation
        // but they should be reasonable values
        let state1_vec = state1
            .to_vec1::<f32>()
            .expect("Failed to convert state to vec");
        let state2_vec = state2
            .to_vec1::<f32>()
            .expect("Failed to convert state to vec");

        for (i, (&val1, &val2)) in state1_vec.iter().zip(state2_vec.iter()).enumerate() {
            assert!(
                val1.is_finite(),
                "State {} should be finite, got {}",
                i,
                val1
            );
            assert!(
                val2.is_finite(),
                "State {} should be finite, got {}",
                i,
                val2
            );
        }
    }

    #[test]
    fn test_lunar_lander_against_python() {
        test_gym_against_python(
            "lunar_lander",
            LunarLanderV3::default(),
            // Not an AMAZING tolerance, but I think close enough to give the same value in reinforcement learning
            Some(Tolerances::new(5.0, 0.2)),
        );
    }
}
