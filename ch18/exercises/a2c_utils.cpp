// C++ port of ch18/exercises/a2c_utils.py (LunarLanderContinuous on Box2D)
#include "a2c_utils.hpp"

#include <pora/core.hpp>

#include <box2d/box2d.h>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>

namespace {

constexpr double FPS = 50;
constexpr double SCALE = 30.0;  // affects how fast-paced the game is, forces should be adjusted as well

constexpr double MAIN_ENGINE_POWER = 13.0;
constexpr double SIDE_ENGINE_POWER = 0.6;

constexpr double INITIAL_RANDOM = 1000.0;  // Set 1500 to make game harder

const std::vector<std::pair<double, double>> LANDER_POLY = {{-14, +17}, {-17, 0}, {-17, -10},
                                                            {+17, -10}, {+17, 0}, {+14, +17}};
constexpr double LEG_AWAY = 20;
constexpr double LEG_DOWN = 18;
constexpr double LEG_W = 2, LEG_H = 8;
constexpr double LEG_SPRING_TORQUE = 40;

constexpr double SIDE_ENGINE_HEIGHT = 14;
constexpr double SIDE_ENGINE_AWAY = 12;
constexpr double MAIN_ENGINE_Y_LOCATION = 4;  // The Y location of the main engine on the body of the Lander.

constexpr int VIEWPORT_W = 600;
constexpr int VIEWPORT_H = 400;

constexpr double GRAVITY = -10.0;

}  // namespace

class LunarLanderContactDetector : public b2ContactListener {
 public:
  explicit LunarLanderContactDetector(LunarLanderEnv* env) : env_(env) {}

  void BeginContact(b2Contact* contact) override {
    b2Body* a = contact->GetFixtureA()->GetBody();
    b2Body* b = contact->GetFixtureB()->GetBody();
    if (env_->lander == a || env_->lander == b) env_->game_over = true;
    for (int i = 0; i < 2; ++i)
      if (env_->legs[i] == a || env_->legs[i] == b) env_->leg_contact[i] = true;
  }

  void EndContact(b2Contact* contact) override {
    b2Body* a = contact->GetFixtureA()->GetBody();
    b2Body* b = contact->GetFixtureB()->GetBody();
    for (int i = 0; i < 2; ++i)
      if (env_->legs[i] == a || env_->legs[i] == b) env_->leg_contact[i] = false;
  }

 private:
  LunarLanderEnv* env_;
};

LunarLanderEnv::LunarLanderEnv() = default;

LunarLanderEnv::~LunarLanderEnv() { destroy(); }

void LunarLanderEnv::destroy() {
  if (!world_) return;
  world_->SetContactListener(nullptr);
  world_.reset();  // destroys all bodies and joints
  moon_ = lander = legs[0] = legs[1] = nullptr;
}

std::vector<float> LunarLanderEnv::reset(long seed) {
  if (seed >= 0) np_random_.seed(static_cast<unsigned long long>(seed));
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  auto uniform = [&](double lo, double hi) { return lo + (hi - lo) * unit(np_random_); };

  destroy();
  world_ = std::make_unique<b2World>(b2Vec2(0, GRAVITY));
  contact_listener_ = std::make_unique<LunarLanderContactDetector>(this);
  world_->SetContactListener(contact_listener_.get());
  game_over = false;
  has_prev_shaping_ = false;
  leg_contact[0] = leg_contact[1] = false;
  elapsed_steps_ = 0;

  const double W = VIEWPORT_W / SCALE;
  const double H = VIEWPORT_H / SCALE;

  // terrain
  const int CHUNKS = 11;
  std::vector<double> height(CHUNKS + 1);
  for (auto& h : height) h = uniform(0, H / 2);
  std::vector<double> chunk_x(CHUNKS);
  for (int i = 0; i < CHUNKS; ++i) chunk_x[i] = W / (CHUNKS - 1) * i;
  helipad_x1_ = chunk_x[CHUNKS / 2 - 1];
  helipad_x2_ = chunk_x[CHUNKS / 2 + 1];
  helipad_y_ = H / 4;
  for (int k = -2; k <= 2; ++k) height[CHUNKS / 2 + k] = helipad_y_;
  std::vector<double> smooth_y(CHUNKS);
  for (int i = 0; i < CHUNKS; ++i)
    smooth_y[i] = 0.33 * (height[(i - 1 + CHUNKS + 1) % (CHUNKS + 1)] + height[i] + height[i + 1]);

  b2BodyDef moon_def;
  moon_ = world_->CreateBody(&moon_def);
  {
    b2EdgeShape edge;
    edge.SetTwoSided(b2Vec2(0, 0), b2Vec2(static_cast<float>(W), 0));
    moon_->CreateFixture(&edge, 0.0f);
  }
  sky_polys_.clear();
  for (int i = 0; i < CHUNKS - 1; ++i) {
    b2Vec2 p1(static_cast<float>(chunk_x[i]), static_cast<float>(smooth_y[i]));
    b2Vec2 p2(static_cast<float>(chunk_x[i + 1]), static_cast<float>(smooth_y[i + 1]));
    b2EdgeShape edge;
    edge.SetTwoSided(p1, p2);
    b2FixtureDef fd;
    fd.shape = &edge;
    fd.density = 0;
    fd.friction = 0.1f;
    moon_->CreateFixture(&fd);
    sky_polys_.push_back({{p1.x, p1.y}, {p2.x, p2.y}, {p2.x, static_cast<float>(H)}, {p1.x, static_cast<float>(H)}});
  }

  // lander
  const double initial_y = VIEWPORT_H / SCALE;
  const double initial_x = VIEWPORT_W / SCALE / 2;
  b2BodyDef lander_def;
  lander_def.type = b2_dynamicBody;
  lander_def.position.Set(static_cast<float>(initial_x), static_cast<float>(initial_y));
  lander_def.angle = 0.0f;
  lander = world_->CreateBody(&lander_def);
  {
    b2Vec2 verts[6];
    for (int i = 0; i < 6; ++i)
      verts[i].Set(static_cast<float>(LANDER_POLY[i].first / SCALE), static_cast<float>(LANDER_POLY[i].second / SCALE));
    b2PolygonShape poly;
    poly.Set(verts, 6);
    b2FixtureDef fd;
    fd.shape = &poly;
    fd.density = 5.0f;
    fd.friction = 0.1f;
    fd.filter.categoryBits = 0x0010;
    fd.filter.maskBits = 0x001;  // collide only with ground
    fd.restitution = 0.0f;
    lander->CreateFixture(&fd);
  }
  lander->ApplyForceToCenter(b2Vec2(static_cast<float>(uniform(-INITIAL_RANDOM, INITIAL_RANDOM)),
                                    static_cast<float>(uniform(-INITIAL_RANDOM, INITIAL_RANDOM))),
                             true);

  // legs
  for (int k = 0; k < 2; ++k) {
    const int i = k == 0 ? -1 : +1;
    b2BodyDef leg_def;
    leg_def.type = b2_dynamicBody;
    leg_def.position.Set(static_cast<float>(initial_x - i * LEG_AWAY / SCALE), static_cast<float>(initial_y));
    leg_def.angle = static_cast<float>(i * 0.05);
    b2Body* leg = world_->CreateBody(&leg_def);
    b2PolygonShape box;
    box.SetAsBox(static_cast<float>(LEG_W / SCALE), static_cast<float>(LEG_H / SCALE));
    b2FixtureDef fd;
    fd.shape = &box;
    fd.density = 1.0f;
    fd.restitution = 0.0f;
    fd.filter.categoryBits = 0x0020;
    fd.filter.maskBits = 0x001;
    leg->CreateFixture(&fd);

    b2RevoluteJointDef rjd;
    rjd.bodyA = lander;
    rjd.bodyB = leg;
    rjd.localAnchorA.Set(0, 0);
    rjd.localAnchorB.Set(static_cast<float>(i * LEG_AWAY / SCALE), static_cast<float>(LEG_DOWN / SCALE));
    rjd.enableMotor = true;
    rjd.enableLimit = true;
    rjd.maxMotorTorque = static_cast<float>(LEG_SPRING_TORQUE);
    rjd.motorSpeed = static_cast<float>(+0.3 * i);  // low enough not to jump back into the sky
    if (i == -1) {
      rjd.lowerAngle = +0.9f - 0.5f;  // The most esoteric numbers here, angled legs have freedom to travel within
      rjd.upperAngle = +0.9f;
    } else {
      rjd.lowerAngle = -0.9f;
      rjd.upperAngle = -0.9f + 0.5f;
    }
    world_->CreateJoint(&rjd);
    legs[k] = leg;
  }

  return step({0.f, 0.f}).observation;
}

LunarLanderStep LunarLanderEnv::step(const std::vector<float>& action_in) {
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  auto uniform = [&](double lo, double hi) { return lo + (hi - lo) * unit(np_random_); };

  const double action[2] = {std::clamp<double>(action_in[0], -1, 1), std::clamp<double>(action_in[1], -1, 1)};

  // Engines
  const double angle = lander->GetAngle();
  const double tip[2] = {std::sin(angle), std::cos(angle)};
  const double side[2] = {-tip[1], tip[0]};
  const double dispersion[2] = {uniform(-1.0, +1.0) / SCALE, uniform(-1.0, +1.0) / SCALE};

  double m_power = 0.0;
  if (action[0] > 0.0) {
    // Main engine
    m_power = (std::clamp(action[0], 0.0, 1.0) + 1.0) * 0.5;  // 0.5..1.0
    const double ox = tip[0] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0]) + side[0] * dispersion[1];
    const double oy = -tip[1] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0]) - side[1] * dispersion[1];
    const b2Vec2 pos = lander->GetPosition();
    const b2Vec2 impulse_pos(static_cast<float>(pos.x + ox), static_cast<float>(pos.y + oy));
    lander->ApplyLinearImpulse(b2Vec2(static_cast<float>(-ox * MAIN_ENGINE_POWER * m_power),
                                      static_cast<float>(-oy * MAIN_ENGINE_POWER * m_power)),
                               impulse_pos, true);
  }

  double s_power = 0.0, direction = 0.0;
  if (std::abs(action[1]) > 0.5) {
    // Orientation/Side engines
    direction = action[1] > 0 ? 1.0 : -1.0;
    s_power = std::clamp(std::abs(action[1]), 0.5, 1.0);
    const double ox = tip[0] * dispersion[0] + side[0] * (3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE);
    const double oy = -tip[1] * dispersion[0] - side[1] * (3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE);
    const b2Vec2 pos = lander->GetPosition();
    const b2Vec2 impulse_pos(static_cast<float>(pos.x + ox - tip[0] * 17 / SCALE),
                             static_cast<float>(pos.y + oy + tip[1] * SIDE_ENGINE_HEIGHT / SCALE));
    lander->ApplyLinearImpulse(b2Vec2(static_cast<float>(-ox * SIDE_ENGINE_POWER * s_power),
                                      static_cast<float>(-oy * SIDE_ENGINE_POWER * s_power)),
                               impulse_pos, true);
  }
  last_m_power_ = m_power;
  last_s_power_ = s_power;
  last_s_dir_ = direction;

  world_->Step(static_cast<float>(1.0 / FPS), 6 * 30, 2 * 30);

  const b2Vec2 pos = lander->GetPosition();
  const b2Vec2 vel = lander->GetLinearVelocity();
  std::vector<float> state = {
      static_cast<float>((pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2)),
      static_cast<float>((pos.y - (helipad_y_ + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2)),
      static_cast<float>(vel.x * (VIEWPORT_W / SCALE / 2) / FPS),
      static_cast<float>(vel.y * (VIEWPORT_H / SCALE / 2) / FPS),
      static_cast<float>(lander->GetAngle()),
      static_cast<float>(20.0 * lander->GetAngularVelocity() / FPS),
      leg_contact[0] ? 1.0f : 0.0f,
      leg_contact[1] ? 1.0f : 0.0f,
  };

  double reward = 0;
  const double shaping = -100 * std::sqrt(state[0] * state[0] + state[1] * state[1]) -
                         100 * std::sqrt(state[2] * state[2] + state[3] * state[3]) - 100 * std::abs(state[4]) +
                         10 * state[6] + 10 * state[7];  // And ten points for legs contact
  if (has_prev_shaping_) reward = shaping - prev_shaping_;
  prev_shaping_ = shaping;
  has_prev_shaping_ = true;

  reward -= m_power * 0.30;  // less fuel spent is better, about -30 for heuristic landing
  reward -= s_power * 0.03;

  bool terminated = false;
  if (game_over || std::abs(state[0]) >= 1.0) {
    terminated = true;
    reward = -100;
  }
  if (!lander->IsAwake()) {
    terminated = true;
    reward = +100;
  }
  ++elapsed_steps_;
  const bool truncated = elapsed_steps_ > max_episode_steps;  // TimeLimit (the reset() step is not counted)
  return {state, reward, terminated, truncated};
}

cv::Mat LunarLanderEnv::render() const {
  cv::Mat surf(VIEWPORT_H, VIEWPORT_W, CV_8UC3, cv::Scalar(255, 255, 255));
  auto P = [&](double x, double y) {  // world (meters) -> pixel, y up (gymnasium flips the image)
    return cv::Point(static_cast<int>(std::lround(x * SCALE)), VIEWPORT_H - 1 - static_cast<int>(std::lround(y * SCALE)));
  };
  for (const auto& p : sky_polys_) {
    std::vector<cv::Point> poly;
    for (const auto& c : p) poly.push_back(P(c.first, c.second));
    cv::fillConvexPoly(surf, poly, cv::Scalar(0, 0, 0), cv::LINE_AA);
  }

  // Engine flames (the particles of the Python version are purely visual)
  if (lander) {
    const double a = lander->GetAngle();
    const b2Vec2 pos = lander->GetPosition();
    if (last_m_power_ > 0) {
      cv::Point base = P(pos.x + std::sin(a) * MAIN_ENGINE_Y_LOCATION / SCALE * 3, pos.y - std::cos(a) * MAIN_ENGINE_Y_LOCATION / SCALE * 3);
      cv::circle(surf, base, static_cast<int>(4 + 6 * last_m_power_), cv::Scalar(51, 128, 255), cv::FILLED, cv::LINE_AA);
    }
    if (last_s_power_ > 0) {
      const double sx = pos.x - std::cos(a) * last_s_dir_ * 20 / SCALE + std::sin(a) * 2 / SCALE;
      const double sy = pos.y - std::sin(a) * last_s_dir_ * 20 / SCALE + std::cos(a) * SIDE_ENGINE_HEIGHT / SCALE;
      cv::circle(surf, P(sx, sy), static_cast<int>(3 + 3 * last_s_power_), cv::Scalar(51, 128, 255), cv::FILLED, cv::LINE_AA);
    }
  }

  // Lander and legs: color1 = RGB(128, 102, 230), color2 = RGB(77, 77, 128)
  const cv::Scalar color1(230, 102, 128), color2(128, 77, 77);
  for (const b2Body* body : {static_cast<const b2Body*>(lander), static_cast<const b2Body*>(legs[0]),
                             static_cast<const b2Body*>(legs[1])}) {
    if (!body) continue;
    for (const b2Fixture* f = body->GetFixtureList(); f; f = f->GetNext()) {
      if (f->GetType() != b2Shape::e_polygon) continue;
      const auto* shape = static_cast<const b2PolygonShape*>(f->GetShape());
      std::vector<cv::Point> path;
      for (int i = 0; i < shape->m_count; ++i) {
        b2Vec2 v = b2Mul(body->GetTransform(), shape->m_vertices[i]);
        path.push_back(P(v.x, v.y));
      }
      cv::fillConvexPoly(surf, path, color1, cv::LINE_AA);
      cv::polylines(surf, path, true, color2, 1, cv::LINE_AA);
    }
  }

  // Flags on the helipad
  for (double x : {helipad_x1_, helipad_x2_}) {
    const double flagy1 = helipad_y_, flagy2 = flagy1 + 50 / SCALE;
    cv::line(surf, P(x, flagy1), P(x, flagy2), cv::Scalar(255, 255, 255), 1);
    std::vector<cv::Point> flag = {P(x, flagy2), P(x, flagy2 - 10 / SCALE), P(x + 25 / SCALE, flagy2 - 5 / SCALE)};
    cv::fillConvexPoly(surf, flag, cv::Scalar(0, 204, 204), cv::LINE_AA);
  }
  return surf;
}

LunarLander::LunarLander(const std::string& dir) : video_file_directory(pora::output_path(dir)) {
  std::filesystem::create_directories(video_file_directory);

  // Create environment
  env.reset(0);
  state_dim = env.observation_dim;
  action_dim = env.action_dim;
}
