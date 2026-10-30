#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <geometry_msgs/msg/twist.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>

namespace
{
constexpr double kPi = 3.14159265358979323846;

double normalizeAngle(double angle)
{
    while (angle > kPi) {
        angle -= 2.0 * kPi;
    }
    while (angle < -kPi) {
        angle += 2.0 * kPi;
    }
    return angle;
}

double clampValue(double value, double lower, double upper)
{
    return std::max(lower, std::min(value, upper));
}

struct ScanPoint
{
    double x;
    double y;
};

struct Cone
{
    double x;
    double y;
    std::size_t point_count;
};

struct BoundaryEstimate
{
    double clearance{0.0};
    double heading{0.0};
    std::size_t inlier_count{0};
};

struct FrontObstacle
{
    bool found{false};
    double range{std::numeric_limits<double>::infinity()};
    double angle{0.0};
};

struct RightSectorDistances
{
    double right{std::numeric_limits<double>::infinity()};
    double right_front{std::numeric_limits<double>::infinity()};
    double front{std::numeric_limits<double>::infinity()};
};
}  // namespace

class LidarProcessor : public rclcpp::Node
{
public:
    LidarProcessor()
    : Node("lidar_processor")
    {
        declare_parameter<std::string>("follow_side", "auto");
        declare_parameter<double>("desired_clearance", 0.75);
        declare_parameter<double>("lookahead_distance", 1.20);
        declare_parameter<double>("min_detection_range", 0.20);
        declare_parameter<double>("max_detection_range", 3.00);
        declare_parameter<double>("side_sector_angle_deg", 85.0);
        declare_parameter<double>("front_dead_zone_deg", 5.0);
        declare_parameter<double>("cluster_gap", 0.16);
        declare_parameter<double>("min_cone_width", 0.03);
        declare_parameter<double>("max_cone_width", 0.45);
        declare_parameter<int>("min_cluster_points", 3);
        declare_parameter<double>("base_speed", 0.60);
        declare_parameter<double>("search_speed", 0.55);
        declare_parameter<double>("min_drive_speed", 0.55);
        declare_parameter<double>("lateral_gain", 0.85);
        declare_parameter<double>("heading_gain", 0.80);
        declare_parameter<double>("clearance_deadband", 0.08);
        declare_parameter<double>("curve_enter_heading_deg", 10.0);
        declare_parameter<double>("curve_exit_heading_deg", 4.0);
        declare_parameter<int>("curve_exit_frames", 10);
        declare_parameter<double>("curve_reverse_heading_deg", 10.0);
        declare_parameter<int>("curve_reverse_frames", 3);
        declare_parameter<double>("curve_min_steering", 0.07);
        declare_parameter<double>("curve_steering_update", 0.15);
        declare_parameter<double>("boundary_inlier_tolerance", 0.18);
        declare_parameter<double>("boundary_smoothing", 0.15);
        declare_parameter<double>("max_boundary_distance_jump", 0.45);
        declare_parameter<double>("max_boundary_heading_jump_deg", 50.0);
        declare_parameter<double>("steering_sign", 1.0);
        declare_parameter<double>("max_steering", 0.25);
        declare_parameter<double>("max_toward_boundary_steering", 0.11);
        declare_parameter<double>("min_away_curve_steering", 0.20);
        declare_parameter<double>("minimum_safe_clearance", 0.50);
        declare_parameter<double>("steering_smoothing", 0.25);
        declare_parameter<double>("target_loss_timeout", 1.20);
        declare_parameter<double>("front_stop_distance", 0.55);
        declare_parameter<double>("front_stop_angle_deg", 35.0);
        declare_parameter<double>("front_avoidance_distance", 1.40);
        declare_parameter<double>("front_avoidance_center_angle_deg", 15.0);
        declare_parameter<double>("front_avoidance_steering", 0.25);
        declare_parameter<double>("front_avoidance_hold_time", 1.50);
        declare_parameter<double>("front_avoidance_max_duration", 5.00);
        declare_parameter<double>("front_emergency_distance", 0.45);
        declare_parameter<double>("front_emergency_angle_deg", 4.0);
        declare_parameter<double>("followed_cone_match_distance", 0.40);
        declare_parameter<double>("local_turn_distance", 0.75);
        declare_parameter<double>("local_turn_release_distance", 0.90);
        declare_parameter<int>("local_turn_confirm_frames", 2);
        declare_parameter<int>("local_turn_release_frames", 3);
        declare_parameter<double>("local_front_sector_deg", 18.0);
        declare_parameter<double>("local_right_front_sector_deg", 55.0);
        declare_parameter<double>("right_turn_far_margin", 0.20);
        declare_parameter<double>("right_turn_preview_distance", 1.20);
        declare_parameter<double>("right_turn_preview_release_distance", 1.35);
        declare_parameter<double>("right_turn_detect_steering", 0.03);
        declare_parameter<double>("critical_turn_speed", 0.50);
        declare_parameter<double>("front_escape_stop_distance", 0.25);
        declare_parameter<double>("left_exit_peak_heading_deg", 20.0);
        declare_parameter<double>("left_exit_heading_deg", 10.0);
        declare_parameter<int>("left_exit_confirm_frames", 3);
        declare_parameter<double>("right_turn_reentry_cooldown", 0.80);

        follow_side_ = get_parameter("follow_side").as_string();
        desired_clearance_ = get_parameter("desired_clearance").as_double();
        lookahead_distance_ = get_parameter("lookahead_distance").as_double();
        min_detection_range_ = get_parameter("min_detection_range").as_double();
        max_detection_range_ = get_parameter("max_detection_range").as_double();
        side_sector_angle_ = get_parameter("side_sector_angle_deg").as_double() * kPi / 180.0;
        front_dead_zone_ = get_parameter("front_dead_zone_deg").as_double() * kPi / 180.0;
        cluster_gap_ = get_parameter("cluster_gap").as_double();
        min_cone_width_ = get_parameter("min_cone_width").as_double();
        max_cone_width_ = get_parameter("max_cone_width").as_double();
        min_cluster_points_ = get_parameter("min_cluster_points").as_int();
        base_speed_ = get_parameter("base_speed").as_double();
        search_speed_ = get_parameter("search_speed").as_double();
        min_drive_speed_ = get_parameter("min_drive_speed").as_double();
        lateral_gain_ = get_parameter("lateral_gain").as_double();
        heading_gain_ = get_parameter("heading_gain").as_double();
        clearance_deadband_ = get_parameter("clearance_deadband").as_double();
        curve_enter_heading_ =
            get_parameter("curve_enter_heading_deg").as_double() * kPi / 180.0;
        curve_exit_heading_ =
            get_parameter("curve_exit_heading_deg").as_double() * kPi / 180.0;
        curve_exit_frames_ = get_parameter("curve_exit_frames").as_int();
        curve_reverse_heading_ =
            get_parameter("curve_reverse_heading_deg").as_double() * kPi / 180.0;
        curve_reverse_frames_ = get_parameter("curve_reverse_frames").as_int();
        curve_min_steering_ = get_parameter("curve_min_steering").as_double();
        curve_steering_update_ = get_parameter("curve_steering_update").as_double();
        boundary_inlier_tolerance_ = get_parameter("boundary_inlier_tolerance").as_double();
        boundary_smoothing_ = get_parameter("boundary_smoothing").as_double();
        max_boundary_distance_jump_ = get_parameter("max_boundary_distance_jump").as_double();
        max_boundary_heading_jump_ =
            get_parameter("max_boundary_heading_jump_deg").as_double() * kPi / 180.0;
        steering_sign_ = get_parameter("steering_sign").as_double();
        max_steering_ = get_parameter("max_steering").as_double();
        max_toward_boundary_steering_ =
            get_parameter("max_toward_boundary_steering").as_double();
        min_away_curve_steering_ =
            get_parameter("min_away_curve_steering").as_double();
        minimum_safe_clearance_ = get_parameter("minimum_safe_clearance").as_double();
        steering_smoothing_ = get_parameter("steering_smoothing").as_double();
        target_loss_timeout_ = get_parameter("target_loss_timeout").as_double();
        front_stop_distance_ = get_parameter("front_stop_distance").as_double();
        front_stop_angle_ = get_parameter("front_stop_angle_deg").as_double() * kPi / 180.0;
        front_avoidance_distance_ = get_parameter("front_avoidance_distance").as_double();
        front_avoidance_center_angle_ =
            get_parameter("front_avoidance_center_angle_deg").as_double() * kPi / 180.0;
        front_avoidance_steering_ = get_parameter("front_avoidance_steering").as_double();
        front_avoidance_hold_time_ =
            get_parameter("front_avoidance_hold_time").as_double();
        front_avoidance_max_duration_ =
            get_parameter("front_avoidance_max_duration").as_double();
        front_emergency_distance_ = get_parameter("front_emergency_distance").as_double();
        front_emergency_angle_ =
            get_parameter("front_emergency_angle_deg").as_double() * kPi / 180.0;
        followed_cone_match_distance_ = get_parameter("followed_cone_match_distance").as_double();
        local_turn_distance_ = get_parameter("local_turn_distance").as_double();
        local_turn_release_distance_ =
            get_parameter("local_turn_release_distance").as_double();
        local_turn_confirm_frames_ = get_parameter("local_turn_confirm_frames").as_int();
        local_turn_release_frames_ = get_parameter("local_turn_release_frames").as_int();
        local_front_sector_angle_ =
            get_parameter("local_front_sector_deg").as_double() * kPi / 180.0;
        local_right_front_sector_angle_ =
            get_parameter("local_right_front_sector_deg").as_double() * kPi / 180.0;
        right_turn_far_margin_ = get_parameter("right_turn_far_margin").as_double();
        right_turn_preview_distance_ =
            get_parameter("right_turn_preview_distance").as_double();
        right_turn_preview_release_distance_ =
            get_parameter("right_turn_preview_release_distance").as_double();
        right_turn_detect_steering_ =
            get_parameter("right_turn_detect_steering").as_double();
        critical_turn_speed_ = get_parameter("critical_turn_speed").as_double();
        front_escape_stop_distance_ =
            get_parameter("front_escape_stop_distance").as_double();
        left_exit_peak_heading_ =
            get_parameter("left_exit_peak_heading_deg").as_double() * kPi / 180.0;
        left_exit_heading_ =
            get_parameter("left_exit_heading_deg").as_double() * kPi / 180.0;
        left_exit_confirm_frames_ = get_parameter("left_exit_confirm_frames").as_int();
        right_turn_reentry_cooldown_ =
            get_parameter("right_turn_reentry_cooldown").as_double();

        desired_clearance_ = std::max(0.10, desired_clearance_);
        lookahead_distance_ = std::max(0.25, lookahead_distance_);
        min_detection_range_ = std::max(0.05, min_detection_range_);
        max_detection_range_ = std::max(min_detection_range_ + 0.10, max_detection_range_);
        cluster_gap_ = std::max(0.02, cluster_gap_);
        max_cone_width_ = std::max(min_cone_width_, max_cone_width_);
        min_drive_speed_ = std::max(0.0, min_drive_speed_);
        base_speed_ = std::max(min_drive_speed_, base_speed_);
        search_speed_ = clampValue(search_speed_, min_drive_speed_, base_speed_);
        clearance_deadband_ = std::max(0.01, clearance_deadband_);
        curve_enter_heading_ = clampValue(curve_enter_heading_, 2.0 * kPi / 180.0, kPi / 2.0);
        curve_exit_heading_ = clampValue(curve_exit_heading_, 1.0 * kPi / 180.0, curve_enter_heading_);
        curve_exit_frames_ = std::max(1, curve_exit_frames_);
        curve_reverse_heading_ = clampValue(
            curve_reverse_heading_, curve_exit_heading_, kPi / 2.0);
        curve_reverse_frames_ = std::max(1, curve_reverse_frames_);
        curve_min_steering_ = std::max(0.01, std::abs(curve_min_steering_));
        curve_steering_update_ = clampValue(curve_steering_update_, 0.01, 1.0);
        boundary_inlier_tolerance_ = std::max(0.05, boundary_inlier_tolerance_);
        boundary_smoothing_ = clampValue(boundary_smoothing_, 0.0, 1.0);
        max_boundary_distance_jump_ = std::max(0.10, max_boundary_distance_jump_);
        max_boundary_heading_jump_ = clampValue(max_boundary_heading_jump_, 5.0 * kPi / 180.0, kPi / 2.0);
        max_steering_ = std::max(0.05, std::abs(max_steering_));
        max_toward_boundary_steering_ = clampValue(
            std::abs(max_toward_boundary_steering_), 0.01, max_steering_);
        min_away_curve_steering_ = clampValue(
            std::abs(min_away_curve_steering_), 0.01, max_steering_);
        minimum_safe_clearance_ = clampValue(
            minimum_safe_clearance_, 0.10, std::max(0.10, desired_clearance_ - 0.05));
        steering_smoothing_ = clampValue(steering_smoothing_, 0.0, 1.0);
        target_loss_timeout_ = std::max(0.0, target_loss_timeout_);
        front_stop_distance_ = std::max(0.0, front_stop_distance_);
        front_stop_angle_ = clampValue(front_stop_angle_, 0.0, kPi / 2.0);
        front_avoidance_distance_ =
            std::max(front_stop_distance_, front_avoidance_distance_);
        front_avoidance_center_angle_ = clampValue(
            front_avoidance_center_angle_, front_emergency_angle_, front_stop_angle_);
        front_avoidance_steering_ = clampValue(
            std::abs(front_avoidance_steering_), 0.01, max_steering_);
        front_avoidance_hold_time_ = std::max(0.10, front_avoidance_hold_time_);
        front_avoidance_max_duration_ =
            std::max(front_avoidance_hold_time_, front_avoidance_max_duration_);
        front_emergency_distance_ = clampValue(
            front_emergency_distance_, 0.0, front_stop_distance_);
        front_emergency_angle_ = clampValue(front_emergency_angle_, 0.0, front_stop_angle_);
        followed_cone_match_distance_ = std::max(0.10, followed_cone_match_distance_);
        local_turn_distance_ = std::max(0.20, local_turn_distance_);
        local_turn_release_distance_ =
            std::max(local_turn_distance_ + 0.05, local_turn_release_distance_);
        local_turn_confirm_frames_ = std::max(1, local_turn_confirm_frames_);
        local_turn_release_frames_ = std::max(1, local_turn_release_frames_);
        local_front_sector_angle_ = clampValue(
            local_front_sector_angle_, 5.0 * kPi / 180.0, 35.0 * kPi / 180.0);
        local_right_front_sector_angle_ = clampValue(
            local_right_front_sector_angle_,
            local_front_sector_angle_ + 5.0 * kPi / 180.0,
            side_sector_angle_);
        right_turn_far_margin_ = std::max(0.05, right_turn_far_margin_);
        right_turn_preview_distance_ = std::max(
            local_turn_release_distance_ + 0.05, right_turn_preview_distance_);
        right_turn_preview_release_distance_ = std::max(
            right_turn_preview_distance_ + 0.05,
            right_turn_preview_release_distance_);
        right_turn_detect_steering_ = clampValue(
            std::abs(right_turn_detect_steering_), 0.01, max_steering_);
        critical_turn_speed_ = clampValue(
            critical_turn_speed_, 0.0, min_drive_speed_);
        front_escape_stop_distance_ = clampValue(
            front_escape_stop_distance_, min_detection_range_,
            front_emergency_distance_);
        left_exit_peak_heading_ = clampValue(
            left_exit_peak_heading_, 10.0 * kPi / 180.0, 45.0 * kPi / 180.0);
        left_exit_heading_ = clampValue(
            left_exit_heading_, 3.0 * kPi / 180.0,
            left_exit_peak_heading_ - 3.0 * kPi / 180.0);
        left_exit_confirm_frames_ = std::max(1, left_exit_confirm_frames_);
        right_turn_reentry_cooldown_ = clampValue(
            right_turn_reentry_cooldown_, 0.0, 3.0);

        if (follow_side_ != "auto" && follow_side_ != "left" && follow_side_ != "right") {
            RCLCPP_WARN(
                get_logger(), "follow_side=%s 无效，已改为 auto", follow_side_.c_str());
            follow_side_ = "auto";
        }

        active_side_ = follow_side_ == "auto" ? "" : follow_side_;
        scan_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan", rclcpp::SensorDataQoS(),
            std::bind(&LidarProcessor::scanCallback, this, std::placeholders::_1));
        velocity_pub_ = create_publisher<geometry_msgs::msg::Twist>("car_cmd_vel", 10);

        RCLCPP_INFO(
            get_logger(),
            "单侧锥桶巡航已启动：follow_side=%s, 目标侧向距离=%.2fm, "
            "向外转弯释放侧距=%.2fm, 最大转向=%.2f, "
            "向锥桶最大转向=%.2f, 向外弯最低转向=%.2f, "
            "安全侧距=%.2fm, 前方绕行=±%.0fdeg/%.2fm, 中心触发=±%.0fdeg, "
            "绕行转向=%.2f, 保持=%.1fs, 最长=%.1fs, 前方急停=%.2fm, "
            "左弯舵机PWM=%.0f, 转向符号=%.0f, 行驶PWM范围=%.0f~%.0f",
            follow_side_.c_str(), desired_clearance_,
            desired_clearance_ + clearance_deadband_, max_steering_,
            max_toward_boundary_steering_, min_away_curve_steering_,
            minimum_safe_clearance_, front_stop_angle_ * 180.0 / kPi,
            front_avoidance_distance_,
            front_avoidance_center_angle_ * 180.0 / kPi,
            front_avoidance_steering_, front_avoidance_hold_time_,
            front_avoidance_max_duration_,
            front_emergency_distance_,
            1500.0 - front_avoidance_steering_ * 1300.0, steering_sign_,
            1500.0 + min_drive_speed_ * 100.0, 1500.0 + base_speed_ * 100.0);
        RCLCPP_INFO(
            get_logger(),
            "local_distance_turn: front=±%.0fdeg, right_front=-%.0f~-%.0fdeg, "
            "right=-%.0f~-%.0fdeg, trigger=%.2fm, release=%.2fm, confirm=%d",
            local_front_sector_angle_ * 180.0 / kPi,
            local_right_front_sector_angle_ * 180.0 / kPi,
            local_front_sector_angle_ * 180.0 / kPi,
            side_sector_angle_ * 180.0 / kPi,
            local_right_front_sector_angle_ * 180.0 / kPi,
            local_turn_distance_, local_turn_release_distance_,
            local_turn_confirm_frames_);
        RCLCPP_INFO(
            get_logger(),
            "right_turn_closed_loop: far_margin=%.2fm, preview=%.2fm, "
            "preview_release=%.2fm, right_steering_threshold=%.3f, "
            "reentry_cooldown=%.2fs",
            right_turn_far_margin_, right_turn_preview_distance_,
            right_turn_preview_release_distance_, right_turn_detect_steering_,
            right_turn_reentry_cooldown_);
        RCLCPP_INFO(
            get_logger(),
            "close_clearance_escape: start=%.2fm, full_steering=%.2fm, "
            "critical_speed=%.2f, absolute_stop=%.2fm",
            minimum_safe_clearance_,
            std::max(min_detection_range_, minimum_safe_clearance_ * 0.5),
            critical_turn_speed_, front_escape_stop_distance_);
        RCLCPP_INFO(
            get_logger(),
            "left_exit_detection: peak=%.0fdeg, exit=%.0fdeg, confirm=%d",
            left_exit_peak_heading_ * 180.0 / kPi,
            left_exit_heading_ * 180.0 / kPi, left_exit_confirm_frames_);
    }

private:
    std::vector<Cone> clusterCones(const std::vector<ScanPoint> & points) const
    {
        std::vector<Cone> cones;
        std::vector<ScanPoint> cluster;

        const auto finish_cluster = [&]() {
            if (cluster.size() < static_cast<std::size_t>(std::max(1, min_cluster_points_))) {
                cluster.clear();
                return;
            }

            double min_x = cluster.front().x;
            double max_x = cluster.front().x;
            double min_y = cluster.front().y;
            double max_y = cluster.front().y;
            double sum_x = 0.0;
            double sum_y = 0.0;
            for (const auto & point : cluster) {
                min_x = std::min(min_x, point.x);
                max_x = std::max(max_x, point.x);
                min_y = std::min(min_y, point.y);
                max_y = std::max(max_y, point.y);
                sum_x += point.x;
                sum_y += point.y;
            }

            const double width = std::hypot(max_x - min_x, max_y - min_y);
            if (width >= min_cone_width_ && width <= max_cone_width_) {
                cones.push_back({
                    sum_x / static_cast<double>(cluster.size()),
                    sum_y / static_cast<double>(cluster.size()),
                    cluster.size()});
            }
            cluster.clear();
        };

        for (const auto & point : points) {
            if (!cluster.empty()) {
                const auto & previous = cluster.back();
                if (std::hypot(point.x - previous.x, point.y - previous.y) > cluster_gap_) {
                    finish_cluster();
                }
            }
            cluster.push_back(point);
        }
        finish_cluster();

        std::sort(cones.begin(), cones.end(), [](const Cone & lhs, const Cone & rhs) {
            return lhs.x < rhs.x;
        });
        return cones;
    }

    void extractCones(
        const sensor_msgs::msg::LaserScan & scan,
        std::vector<Cone> & left_cones,
        std::vector<Cone> & right_cones) const
    {
        std::vector<ScanPoint> left_points;
        std::vector<ScanPoint> right_points;
        left_points.reserve(scan.ranges.size() / 4);
        right_points.reserve(scan.ranges.size() / 4);

        const double lower_range = std::max(min_detection_range_, static_cast<double>(scan.range_min));
        const double upper_range = std::min(max_detection_range_, static_cast<double>(scan.range_max));

        for (std::size_t i = 0; i < scan.ranges.size(); ++i) {
            const double range = scan.ranges[i];
            if (!std::isfinite(range) || range < lower_range || range > upper_range) {
                continue;
            }

            const double angle = normalizeAngle(
                static_cast<double>(scan.angle_min) +
                static_cast<double>(i) * static_cast<double>(scan.angle_increment));
            const double absolute_angle = std::abs(angle);
            if (absolute_angle < front_dead_zone_ || absolute_angle > side_sector_angle_) {
                continue;
            }

            const ScanPoint point{range * std::cos(angle), range * std::sin(angle)};
            if (point.x <= 0.05) {
                continue;
            }

            if (point.y > 0.0) {
                left_points.push_back(point);
            } else {
                right_points.push_back(point);
            }
        }

        left_cones = clusterCones(left_points);
        right_cones = clusterCones(right_points);
    }

    void chooseActiveSide(
        const std::vector<Cone> & left_cones,
        const std::vector<Cone> & right_cones)
    {
        if (!active_side_.empty() || follow_side_ != "auto") {
            return;
        }
        if (left_cones.empty() && right_cones.empty()) {
            return;
        }

        active_side_ = left_cones.size() >= right_cones.size() ? "left" : "right";
        RCLCPP_INFO(
            get_logger(), "自动锁定%s侧锥桶，后续不会在左右两侧间跳变",
            active_side_ == "left" ? "左" : "右");
    }

    bool estimateBoundary(
        const std::vector<Cone> & cones, double side_sign,
        BoundaryEstimate & estimate) const
    {
        if (cones.empty()) {
            return false;
        }

        double best_slope = 0.0;
        double best_intercept = 0.0;
        double best_error = std::numeric_limits<double>::infinity();
        std::size_t best_inlier_count = 0;

        // Try every cone pair and retain the line supported by the most cone centers.
        for (std::size_t first = 0; first < cones.size(); ++first) {
            for (std::size_t second = first + 1; second < cones.size(); ++second) {
                const double delta_x = cones[second].x - cones[first].x;
                if (std::abs(delta_x) < 0.25) {
                    continue;
                }

                const double slope = (cones[second].y - cones[first].y) / delta_x;
                if (std::abs(slope) > 1.0) {
                    continue;
                }
                const double intercept = cones[first].y - slope * cones[first].x;
                const double signed_clearance = side_sign * intercept;
                if (signed_clearance < 0.10 || signed_clearance > 2.50) {
                    continue;
                }
                const double heading = std::atan(slope);
                if (has_boundary_state_ &&
                    (std::abs(signed_clearance - filtered_clearance_) >
                        max_boundary_distance_jump_ ||
                     std::abs(normalizeAngle(heading - filtered_boundary_heading_)) >
                        max_boundary_heading_jump_))
                {
                    continue;
                }

                const double line_scale = std::sqrt(1.0 + slope * slope);
                std::size_t inlier_count = 0;
                double total_error = 0.0;
                for (const auto & cone : cones) {
                    const double error =
                        std::abs(cone.y - (slope * cone.x + intercept)) / line_scale;
                    if (error <= boundary_inlier_tolerance_) {
                        ++inlier_count;
                        total_error += error;
                    }
                }
                if (has_boundary_state_) {
                    total_error +=
                        0.20 * std::abs(signed_clearance - filtered_clearance_) +
                        0.10 * std::abs(normalizeAngle(heading - filtered_boundary_heading_));
                }

                if (inlier_count > best_inlier_count ||
                    (inlier_count == best_inlier_count && total_error < best_error))
                {
                    best_slope = slope;
                    best_intercept = intercept;
                    best_inlier_count = inlier_count;
                    best_error = total_error;
                }
            }
        }

        if (best_inlier_count < 2) {
            const Cone * nearest_lookahead = &cones.front();
            for (const auto & cone : cones) {
                if (std::abs(cone.x - lookahead_distance_) <
                    std::abs(nearest_lookahead->x - lookahead_distance_))
                {
                    nearest_lookahead = &cone;
                }
            }
            estimate.clearance = std::abs(nearest_lookahead->y);
            estimate.heading = has_boundary_state_ ? filtered_boundary_heading_ : 0.0;
            estimate.inlier_count = 1;
            return estimate.clearance >= 0.10 && estimate.clearance <= 2.50;
        }

        // Refit the selected inliers using least squares for a less noisy heading.
        std::vector<const Cone *> inliers;
        const double best_line_scale = std::sqrt(1.0 + best_slope * best_slope);
        for (const auto & cone : cones) {
            const double error =
                std::abs(cone.y - (best_slope * cone.x + best_intercept)) / best_line_scale;
            if (error <= boundary_inlier_tolerance_) {
                inliers.push_back(&cone);
            }
        }

        double mean_x = 0.0;
        double mean_y = 0.0;
        for (const Cone * cone : inliers) {
            mean_x += cone->x;
            mean_y += cone->y;
        }
        mean_x /= static_cast<double>(inliers.size());
        mean_y /= static_cast<double>(inliers.size());

        double covariance = 0.0;
        double variance_x = 0.0;
        for (const Cone * cone : inliers) {
            covariance += (cone->x - mean_x) * (cone->y - mean_y);
            variance_x += (cone->x - mean_x) * (cone->x - mean_x);
        }
        if (variance_x > 1e-6) {
            best_slope = clampValue(covariance / variance_x, -1.0, 1.0);
            best_intercept = mean_y - best_slope * mean_x;
        }

        estimate.clearance = std::abs(best_intercept);
        estimate.heading = std::atan(best_slope);
        estimate.inlier_count = inliers.size();
        if (side_sign * best_intercept <= 0.0) {
            return false;
        }
        if (has_boundary_state_ &&
            (std::abs(estimate.clearance - filtered_clearance_) >
                max_boundary_distance_jump_ ||
             std::abs(normalizeAngle(estimate.heading - filtered_boundary_heading_)) >
                max_boundary_heading_jump_))
        {
            return false;
        }
        return true;
    }

    bool calculateSteering(
        const std::vector<Cone> & cones, double side_sign, double & steering,
        BoundaryEstimate & raw_estimate, bool front_turn_required)
    {
        if (!estimateBoundary(cones, side_sign, raw_estimate)) {
            return false;
        }

        if (has_boundary_state_ &&
            std::abs(raw_estimate.clearance - filtered_clearance_) >
                max_boundary_distance_jump_)
        {
            return false;
        }

        if (!has_boundary_state_) {
            filtered_clearance_ = raw_estimate.clearance;
            filtered_boundary_heading_ = raw_estimate.heading;
            has_boundary_state_ = true;
        } else {
            filtered_clearance_ =
                boundary_smoothing_ * raw_estimate.clearance +
                (1.0 - boundary_smoothing_) * filtered_clearance_;
            filtered_boundary_heading_ =
                boundary_smoothing_ * raw_estimate.heading +
                (1.0 - boundary_smoothing_) * filtered_boundary_heading_;
        }

        const double clearance_error = desired_clearance_ - filtered_clearance_;
        double lateral_steering =
            steering_sign_ * (-side_sign * lateral_gain_ * clearance_error);
        const double heading_steering =
            steering_sign_ * heading_gain_ * filtered_boundary_heading_;
        clearance_safety_active_ = false;

        // Limit only the distance correction toward the boundary. The boundary
        // heading is added afterwards so a confirmed bend cannot be hidden by
        // a large clearance error.
        const double steering_polarity = steering_sign_ >= 0.0 ? 1.0 : -1.0;
        const double toward_boundary_direction = side_sign * steering_polarity;
        const double away_boundary_direction = -toward_boundary_direction;
        const bool heading_requests_away =
            heading_steering * away_boundary_direction > 0.0;
        const bool boundary_is_too_far =
            filtered_clearance_ > desired_clearance_ + clearance_deadband_;
        const bool hold_toward_boundary =
            heading_requests_away && boundary_is_too_far && !front_turn_required;
        if (lateral_steering * toward_boundary_direction >
            max_toward_boundary_steering_)
        {
            lateral_steering =
                toward_boundary_direction * max_toward_boundary_steering_;
        }

        if (!curve_mode_ && !hold_toward_boundary &&
            std::abs(filtered_boundary_heading_) >= curve_enter_heading_ &&
            std::abs(heading_steering) > 1e-6)
        {
            curve_mode_ = true;
            curve_direction_ = heading_steering > 0.0 ? 1.0 : -1.0;
            curve_base_steering_ = clampValue(
                heading_steering, -max_steering_, max_steering_);
            if (std::abs(curve_base_steering_) < curve_min_steering_) {
                curve_base_steering_ = curve_direction_ * curve_min_steering_;
            }
            curve_exit_count_ = 0;
            curve_reverse_count_ = 0;
            RCLCPP_INFO(
                get_logger(), "进入大弯保持模式：方向=%s, 基础转向=%.3f",
                curve_direction_ > 0.0 ? "左" : "右", curve_base_steering_);
        }

        if (curve_mode_) {
            // Leave curve mode based on boundary shape. Clearance correction
            // continues in normal mode and must not keep an old turn latched.
            const bool boundary_is_straight =
                raw_estimate.inlier_count >= 2 &&
                std::abs(raw_estimate.heading) <= curve_exit_heading_;
            curve_exit_count_ = boundary_is_straight ? curve_exit_count_ + 1 : 0;
            if (curve_exit_count_ >= curve_exit_frames_) {
                curve_mode_ = false;
                curve_base_steering_ = 0.0;
                curve_direction_ = 0.0;
                curve_exit_count_ = 0;
                curve_reverse_count_ = 0;
                RCLCPP_INFO(get_logger(), "连续确认边界变直，退出大弯保持模式");
            }
        }

        if (curve_mode_) {
            const double observed_direction = heading_steering >= 0.0 ? 1.0 : -1.0;
            const bool reversing_from_toward_to_away =
                curve_direction_ * toward_boundary_direction > 0.0 &&
                observed_direction * away_boundary_direction > 0.0;
            const bool delay_away_reversal =
                reversing_from_toward_to_away && boundary_is_too_far &&
                !front_turn_required;
            const bool opposite_curve_observed =
                raw_estimate.inlier_count >= 2 &&
                std::abs(filtered_boundary_heading_) >= curve_reverse_heading_ &&
                observed_direction * curve_direction_ < 0.0 &&
                !delay_away_reversal;

            if (opposite_curve_observed) {
                ++curve_reverse_count_;
                // Release the stale turn while the opposite bend is confirmed.
                curve_base_steering_ *= (1.0 - curve_steering_update_);
            } else {
                curve_reverse_count_ = 0;
            }

            if (curve_reverse_count_ >= curve_reverse_frames_) {
                curve_direction_ = observed_direction;
                curve_base_steering_ = curve_direction_ * curve_min_steering_;
                curve_reverse_count_ = 0;
                RCLCPP_INFO(
                    get_logger(), "边界弯向发生变化，大弯基础转向已切换为%s",
                    curve_direction_ > 0.0 ? "左" : "右");
            }

            if (std::abs(filtered_boundary_heading_) > curve_exit_heading_ &&
                observed_direction * curve_direction_ > 0.0)
            {
                double target_curve_steering = clampValue(
                    heading_steering, -max_steering_, max_steering_);
                if (std::abs(target_curve_steering) < curve_min_steering_) {
                    target_curve_steering = curve_direction_ * curve_min_steering_;
                }
                curve_base_steering_ =
                    curve_steering_update_ * target_curve_steering +
                    (1.0 - curve_steering_update_) * curve_base_steering_;
            }

            // Keep following the measured bend while applying the already
            // limited distance correction.
            steering = curve_base_steering_ + lateral_steering;
            if (std::abs(clearance_error) <= clearance_deadband_ &&
                steering * curve_direction_ >= 0.0 &&
                std::abs(steering) < curve_min_steering_)
            {
                steering = curve_direction_ * curve_min_steering_;
            }
        } else {
            steering = lateral_steering + heading_steering;
        }

        // Moving toward the followed cones must be gentle. Moving away can use
        // the full steering range when clearance is already unsafe.
        if (steering * toward_boundary_direction > max_toward_boundary_steering_) {
            steering = toward_boundary_direction * max_toward_boundary_steering_;
        }

        if (hold_toward_boundary) {
            // Do not leave a distant followed boundary just because its fitted
            // heading briefly points away. Approach it gently until there is
            // enough lateral position, or a front cone requires the turn.
            steering = toward_boundary_direction * max_toward_boundary_steering_;
            RCLCPP_INFO_THROTTLE(
                get_logger(), *get_clock(), 1000,
                "跟随侧仍过远(%.2fm)，暂缓向外转弯并继续贴近",
                filtered_clearance_);
        }

        if (curve_mode_ &&
            curve_direction_ * away_boundary_direction > 0.0 &&
            std::abs(filtered_boundary_heading_) >= curve_enter_heading_ &&
            steering * away_boundary_direction > 0.0 &&
            std::abs(steering) < min_away_curve_steering_)
        {
            steering = away_boundary_direction * min_away_curve_steering_;
        }

        // A single fitted line can briefly cut inside the cone row as the
        // fourth/fifth cone enters view. Use the smoothed boundary here; the
        // right-sector scan still provides immediate close-range protection.
        const double safety_clearance = filtered_clearance_;
        if (safety_clearance < minimum_safe_clearance_) {
            const double away_magnitude = clampValue(
                curve_min_steering_ + lateral_gain_ *
                    (minimum_safe_clearance_ - safety_clearance),
                curve_min_steering_, max_steering_);
            steering = -toward_boundary_direction * away_magnitude;
            clearance_safety_active_ = true;
        }

        steering = clampValue(steering, -max_steering_, max_steering_);
        return true;
    }

    RightSectorDistances measureRightSectors(
        const sensor_msgs::msg::LaserScan & scan) const
    {
        std::vector<double> right_samples;
        std::vector<double> right_front_samples;
        std::vector<double> front_samples;

        const double lower_range =
            std::max(min_detection_range_, static_cast<double>(scan.range_min));
        const double upper_range =
            std::min(max_detection_range_, static_cast<double>(scan.range_max));

        for (std::size_t i = 0; i < scan.ranges.size(); ++i) {
            const double range = scan.ranges[i];
            if (!std::isfinite(range) || range < lower_range || range > upper_range) {
                continue;
            }

            const double angle = normalizeAngle(
                static_cast<double>(scan.angle_min) +
                static_cast<double>(i) * static_cast<double>(scan.angle_increment));
            if (std::abs(angle) <= local_front_sector_angle_) {
                front_samples.push_back(range);
            } else if (angle < -local_front_sector_angle_ &&
                angle >= -local_right_front_sector_angle_)
            {
                right_front_samples.push_back(range);
            } else if (angle < -local_right_front_sector_angle_ &&
                angle >= -side_sector_angle_)
            {
                right_samples.push_back(range);
            }
        }

        const auto robust_nearest = [this](std::vector<double> & samples) {
            const std::size_t required_samples = static_cast<std::size_t>(
                std::max(3, min_cluster_points_));
            if (samples.size() < required_samples) {
                return std::numeric_limits<double>::infinity();
            }
            const auto selected = samples.begin() +
                static_cast<std::vector<double>::difference_type>(required_samples - 1);
            std::nth_element(samples.begin(), selected, samples.end());
            return *selected;
        };

        RightSectorDistances distances;
        distances.right = robust_nearest(right_samples);
        distances.right_front = robust_nearest(right_front_samples);
        distances.front = robust_nearest(front_samples);
        return distances;
    }

    void resetRightTurnControl()
    {
        right_turn_approach_active_ = false;
        local_left_turn_active_ = false;
        local_hard_left_turn_ = false;
        clearance_escape_active_ = false;
        local_turn_confirm_count_ = 0;
        local_turn_release_count_ = 0;
        local_left_peak_seen_ = false;
        left_exit_confirm_count_ = 0;
    }

    double closeClearanceRatio(double right_clearance) const
    {
        if (!std::isfinite(right_clearance) ||
            right_clearance >= minimum_safe_clearance_)
        {
            return 0.0;
        }

        const double full_steering_clearance =
            std::max(min_detection_range_, minimum_safe_clearance_ * 0.5);
        const double transition =
            std::max(0.01, minimum_safe_clearance_ - full_steering_clearance);
        return clampValue(
            (minimum_safe_clearance_ - right_clearance) / transition,
            0.0, 1.0);
    }

    double localLeftSteering(double right_clearance) const
    {
        const double ratio = local_hard_left_turn_ ?
            1.0 : closeClearanceRatio(right_clearance);
        if (clearance_escape_active_) {
            return curve_min_steering_ +
                   ratio * (max_steering_ - curve_min_steering_);
        }
        return front_avoidance_steering_ +
               ratio * (max_steering_ - front_avoidance_steering_);
    }

    double localTurnSpeed(double right_clearance) const
    {
        const double ratio = local_hard_left_turn_ ?
            1.0 : closeClearanceRatio(right_clearance);
        return min_drive_speed_ -
               ratio * (min_drive_speed_ - critical_turn_speed_);
    }

    bool canEscapeFollowedSideObstacle(
        const FrontObstacle & obstacle, bool following_left) const
    {
        return !following_left && local_left_turn_active_ && obstacle.found &&
               obstacle.range > front_escape_stop_distance_ &&
               obstacle.angle < -front_emergency_angle_;
    }

    void updateRightTurnControl(
        const RightSectorDistances & distances, double candidate_steering,
        double right_clearance, bool boundary_reliable)
    {
        const auto now = this->now();
        const double since_left_exit = has_left_exit_time_ ?
            (now - last_left_exit_time_).seconds() :
            std::numeric_limits<double>::infinity();
        right_turn_reentry_blocked_ = right_turn_reentry_cooldown_ > 0.0 &&
            since_left_exit >= 0.0 &&
            since_left_exit < right_turn_reentry_cooldown_;

        const double steering_polarity = steering_sign_ >= 0.0 ? 1.0 : -1.0;
        const double right_turn_direction = -steering_polarity;
        const bool turning_right =
            candidate_steering * right_turn_direction >= right_turn_detect_steering_;
        const double far_distance = desired_clearance_ + right_turn_far_margin_;
        const bool fitted_boundary_far =
            has_boundary_state_ && filtered_clearance_ >= far_distance;
        const bool local_boundary_far =
            std::isfinite(distances.right) && distances.right >= far_distance &&
            (!std::isfinite(distances.right_front) ||
             distances.right_front >= far_distance);
        const double local_right_clearance =
            std::min(distances.right, distances.right_front);

        if (!right_turn_reentry_blocked_ &&
            !local_left_turn_active_ && !right_turn_approach_active_ &&
            turning_right && (fitted_boundary_far || local_boundary_far))
        {
            right_turn_approach_active_ = true;
            local_turn_confirm_count_ = 0;
            RCLCPP_INFO(
                get_logger(),
                "进入右弯接近状态：侧距=%.2fm, 右=%.2fm, 右前=%.2fm, 转向=%.3f",
                filtered_clearance_, distances.right,
            distances.right_front, candidate_steering);
        }

        if (local_left_turn_active_ && !clearance_escape_active_) {
            if (filtered_boundary_heading_ >= left_exit_peak_heading_) {
                local_left_peak_seen_ = true;
            }
            const bool left_exit_candidate =
                local_left_peak_seen_ && boundary_reliable &&
                std::abs(filtered_boundary_heading_) <= left_exit_heading_ &&
                right_clearance >= minimum_safe_clearance_ &&
                distances.front > local_turn_distance_;
            left_exit_confirm_count_ = left_exit_candidate ?
                left_exit_confirm_count_ + 1 : 0;
            if (left_exit_confirm_count_ >= left_exit_confirm_frames_) {
                RCLCPP_INFO(
                    get_logger(),
                    "确认左弯出口，交还直线侧距控制：侧距=%.2fm, "
                    "边界方向=%.1fdeg, 前=%.2fm",
                    right_clearance,
                    filtered_boundary_heading_ * 180.0 / kPi,
                    distances.front);
                resetRightTurnControl();
                last_left_exit_time_ = now;
                has_left_exit_time_ = true;
                right_turn_reentry_blocked_ = right_turn_reentry_cooldown_ > 0.0;
                curve_mode_ = false;
                curve_direction_ = 0.0;
                curve_base_steering_ = 0.0;
                curve_exit_count_ = 0;
                curve_reverse_count_ = 0;
                filtered_steering_ = 0.0;
                return;
            }
        }

        const bool front_close = distances.front <= local_turn_distance_;
        const bool corner_close =
            local_left_turn_active_ && !clearance_escape_active_ &&
            distances.right <= local_turn_distance_ &&
            distances.right_front <= local_turn_distance_;
        const bool hard_left = front_close || corner_close;

        if (hard_left) {
            right_turn_approach_active_ = false;
            local_left_turn_active_ = true;
            local_hard_left_turn_ = true;
            clearance_escape_active_ = false;
            local_turn_confirm_count_ = local_turn_confirm_frames_;
            local_turn_release_count_ = 0;
            return;
        }

        if (!local_left_turn_active_ &&
            local_right_clearance <= minimum_safe_clearance_)
        {
            right_turn_approach_active_ = false;
            local_left_turn_active_ = true;
            local_hard_left_turn_ = false;
            clearance_escape_active_ = true;
            local_turn_confirm_count_ = local_turn_confirm_frames_;
            local_turn_release_count_ = 0;
            RCLCPP_INFO(
                get_logger(),
                "右侧安全距离不足，进入渐进侧距避让：最近右侧=%.2fm",
                local_right_clearance);
            return;
        }

        if (clearance_escape_active_) {
            const double clearance_release = std::min(
                desired_clearance_, minimum_safe_clearance_ + clearance_deadband_);
            const bool clearance_observed = boundary_reliable ||
                std::isfinite(distances.right) ||
                std::isfinite(distances.right_front);
            const bool clearance_recovered = clearance_observed &&
                right_clearance >= clearance_release &&
                distances.front > local_turn_distance_;
            local_turn_release_count_ = clearance_recovered ?
                local_turn_release_count_ + 1 : 0;
            if (local_turn_release_count_ >= local_turn_release_frames_) {
                RCLCPP_INFO(
                    get_logger(),
                    "侧距避让完成，交还普通巡航：最近右侧=%.2fm, 释放=%.2fm",
                    right_clearance, clearance_release);
                resetRightTurnControl();
            }
            return;
        }

        const bool preview_left =
            right_turn_approach_active_ &&
            (distances.right_front <= right_turn_preview_distance_ ||
             distances.front <= right_turn_preview_distance_);
        if (!local_left_turn_active_ && preview_left) {
            local_turn_release_count_ = 0;
            ++local_turn_confirm_count_;
            if (local_turn_confirm_count_ >= local_turn_confirm_frames_) {
                right_turn_approach_active_ = false;
                local_left_turn_active_ = true;
                local_hard_left_turn_ = false;
                clearance_escape_active_ = false;
                RCLCPP_INFO(
                    get_logger(),
                    "右弯接近完成，提前左转：右=%.2fm, 右前=%.2fm, 前=%.2fm",
                    distances.right, distances.right_front, distances.front);
            }
            return;
        }

        if (!local_left_turn_active_) {
            if (!preview_left) {
                local_turn_confirm_count_ = 0;
            }
            local_hard_left_turn_ = false;
            local_turn_release_count_ = 0;
            return;
        }

        local_hard_left_turn_ = false;
        const double recovered_side_distance =
            std::min(local_turn_release_distance_, desired_clearance_);
        const bool recovery_still_needed =
            right_clearance <= recovered_side_distance ||
            distances.right_front <= right_turn_preview_release_distance_ ||
            distances.front <= right_turn_preview_release_distance_;
        if (recovery_still_needed) {
            local_turn_release_count_ = 0;
            return;
        }

        ++local_turn_release_count_;
        if (local_turn_release_count_ >= local_turn_release_frames_) {
            RCLCPP_INFO(
                get_logger(),
                "右弯左转恢复完成，交还侧距控制：右=%.2fm, 右前=%.2fm, 前=%.2fm",
                distances.right, distances.right_front, distances.front);
            resetRightTurnControl();
            last_left_exit_time_ = now;
            has_left_exit_time_ = true;
            right_turn_reentry_blocked_ = right_turn_reentry_cooldown_ > 0.0;
            curve_mode_ = false;
            curve_direction_ = 0.0;
            curve_base_steering_ = 0.0;
            curve_exit_count_ = 0;
            curve_reverse_count_ = 0;
            filtered_steering_ = 0.0;
        }
    }

    FrontObstacle findFrontObstacle(const sensor_msgs::msg::LaserScan & scan) const
    {
        FrontObstacle obstacle;
        if (front_stop_distance_ <= 0.0 || front_stop_angle_ <= 0.0) {
            return obstacle;
        }

        for (std::size_t i = 0; i < scan.ranges.size(); ++i) {
            const double range = scan.ranges[i];
            if (!std::isfinite(range) || range < scan.range_min ||
                range > front_avoidance_distance_)
            {
                continue;
            }
            const double angle = normalizeAngle(
                static_cast<double>(scan.angle_min) +
                static_cast<double>(i) * static_cast<double>(scan.angle_increment));
            if (std::abs(angle) <= front_stop_angle_ && range < obstacle.range) {
                obstacle.found = true;
                obstacle.range = range;
                obstacle.angle = angle;
            }
        }
        return obstacle;
    }

    bool frontObstacleIsEmergency(const FrontObstacle & obstacle) const
    {
        return obstacle.found &&
               (obstacle.range <= front_emergency_distance_ ||
                (obstacle.range <= front_stop_distance_ &&
                 std::abs(obstacle.angle) <= front_emergency_angle_));
    }

    bool frontObstacleOnFollowedSide(
        const FrontObstacle & obstacle, bool following_left) const
    {
        if (!obstacle.found) {
            return false;
        }
        if (std::abs(obstacle.angle) <= front_emergency_angle_ &&
            obstacle.range > front_stop_distance_)
        {
            return true;
        }
        return following_left ? obstacle.angle > 0.0 : obstacle.angle < 0.0;
    }

    bool frontObstacleIsFollowedCone(
        const FrontObstacle & obstacle, const std::vector<Cone> & active_cones,
        bool following_left) const
    {
        if (!obstacle.found) {
            return false;
        }
        if ((following_left && obstacle.angle <= 0.0) ||
            (!following_left && obstacle.angle >= 0.0))
        {
            return false;
        }

        const double obstacle_x = obstacle.range * std::cos(obstacle.angle);
        const double obstacle_y = obstacle.range * std::sin(obstacle.angle);
        for (const auto & cone : active_cones) {
            if (std::hypot(cone.x - obstacle_x, cone.y - obstacle_y) <=
                followed_cone_match_distance_)
            {
                return true;
            }
        }
        return false;
    }

    void publishStop(const char * reason)
    {
        geometry_msgs::msg::Twist command;
        velocity_pub_->publish(command);
        filtered_steering_ = 0.0;
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "%s", reason);
    }

    void scanCallback(const sensor_msgs::msg::LaserScan::SharedPtr scan)
    {
        if (scan->ranges.empty() || !std::isfinite(scan->angle_increment) ||
            std::abs(scan->angle_increment) < std::numeric_limits<float>::epsilon())
        {
            publishStop("雷达数据无效，停车");
            return;
        }

        const FrontObstacle front_obstacle = findFrontObstacle(*scan);

        std::vector<Cone> left_cones;
        std::vector<Cone> right_cones;
        extractCones(*scan, left_cones, right_cones);
        chooseActiveSide(left_cones, right_cones);

        if (active_side_.empty()) {
            publishStop("尚未检测到可锁定的单侧锥桶，停车");
            return;
        }

        const bool following_left = active_side_ == "left";
        const auto & active_cones = following_left ? left_cones : right_cones;
        const RightSectorDistances local_distances = measureRightSectors(*scan);
        if (following_left) {
            resetRightTurnControl();
        }
        const bool front_turn_required = following_left ?
            (front_obstacle.found &&
             front_obstacle.range <= front_avoidance_distance_) :
            local_left_turn_active_;
        double steering = 0.0;
        BoundaryEstimate raw_estimate;
        const bool has_target = calculateSteering(
            active_cones, following_left ? 1.0 : -1.0,
            steering, raw_estimate, front_turn_required);
        double right_clearance = std::numeric_limits<double>::infinity();
        const auto include_right_clearance = [&right_clearance](double distance) {
            if (std::isfinite(distance)) {
                right_clearance = std::min(right_clearance, distance);
            }
        };
        if (has_boundary_state_) {
            include_right_clearance(filtered_clearance_);
        }
        include_right_clearance(local_distances.right);
        include_right_clearance(local_distances.right_front);
        if (!following_left) {
            const bool boundary_reliable =
                has_target && raw_estimate.inlier_count >= 2;
            updateRightTurnControl(
                local_distances, has_target ? steering : filtered_steering_,
                right_clearance, boundary_reliable);
        }

        const auto now = this->now();
        const bool front_followed_cone = frontObstacleIsFollowedCone(
            front_obstacle, active_cones, following_left);
        const bool front_on_followed_side =
            frontObstacleOnFollowedSide(front_obstacle, following_left);
        const double steering_polarity = steering_sign_ >= 0.0 ? 1.0 : -1.0;
        const double away_direction =
            following_left ? -steering_polarity : steering_polarity;
        const bool curve_requests_away =
            curve_mode_ && curve_direction_ * away_direction > 0.0;
        const bool obstacle_near_center =
            front_obstacle.found &&
            std::abs(front_obstacle.angle) <= front_avoidance_center_angle_;
        const bool close_followed_obstacle =
            front_obstacle.found &&
            front_obstacle.range <= front_stop_distance_ &&
            (front_on_followed_side || front_followed_cone);
        const bool start_front_avoidance =
            front_obstacle.found &&
            front_obstacle.range <= front_avoidance_distance_ &&
            (curve_requests_away || obstacle_near_center || close_followed_obstacle);

        if (!following_left) {
            // Right-side following uses only the current right/right-front/front distances.
            front_avoidance_mode_ = false;
        } else {
            if (start_front_avoidance) {
                if (!front_avoidance_mode_) {
                    front_avoidance_started_time_ = now;
                }
                front_avoidance_mode_ = true;
                front_avoidance_last_seen_time_ = now;
            } else if (front_avoidance_mode_ && front_obstacle.found &&
                front_obstacle.range <= front_avoidance_distance_)
            {
                // Keep the maneuver latched while the same cone crosses the vehicle
                // centerline from the followed side to the opposite side.
                front_avoidance_last_seen_time_ = now;
            }

            if (front_avoidance_mode_) {
                const double maneuver_seconds =
                    (now - front_avoidance_started_time_).seconds();
                const double unseen_seconds =
                    (now - front_avoidance_last_seen_time_).seconds();
                if (maneuver_seconds > front_avoidance_max_duration_ ||
                    unseen_seconds > front_avoidance_hold_time_)
                {
                    front_avoidance_mode_ = false;
                }
            }
        }
        const bool front_avoidance_active = following_left ?
            front_avoidance_mode_ : local_left_turn_active_;
        const bool apply_front_avoidance_steering =
            front_avoidance_active &&
            (following_left || !clearance_escape_active_);

        if (has_target) {
            last_target_time_ = now;
            has_seen_target_ = true;
            const double steering_alpha = clearance_safety_active_ ?
                std::max(0.65, steering_smoothing_) : steering_smoothing_;
            filtered_steering_ =
                steering_alpha * steering + (1.0 - steering_alpha) * filtered_steering_;

            bool followed_cone_in_front = front_avoidance_active;
            if (front_obstacle.found) {
                const bool emergency = frontObstacleIsEmergency(front_obstacle);
                const bool hard_emergency =
                    front_obstacle.range <= front_emergency_distance_;
                const bool collision_distance =
                    front_obstacle.range <= front_stop_distance_;
                const bool escaping_followed_side =
                    canEscapeFollowedSideObstacle(front_obstacle, following_left);

                if ((hard_emergency && !escaping_followed_side) ||
                    (!front_avoidance_active &&
                     (emergency ||
                      (collision_distance &&
                       !front_on_followed_side && !front_followed_cone))))
                {
                    publishStop("正前方存在碰撞风险，停车");
                    return;
                }

                if (hard_emergency && escaping_followed_side) {
                    RCLCPP_WARN_THROTTLE(
                        get_logger(), *get_clock(), 500,
                        "右前锥桶距离=%.2fm，继续低速增大左转脱险",
                        front_obstacle.range);
                }

                RCLCPP_WARN_THROTTLE(
                    get_logger(), *get_clock(), 500,
                    "前方锥桶：距离=%.2fm, 角度=%.1fdeg，绕行=%s",
                    front_obstacle.range, front_obstacle.angle * 180.0 / kPi,
                    front_avoidance_active ? "是" : "否");
            }

            if (apply_front_avoidance_steering) {
                const double avoidance_target =
                    away_direction * front_avoidance_steering_;
                filtered_steering_ =
                    0.75 * avoidance_target + 0.25 * filtered_steering_;
            }
            if (local_left_turn_active_) {
                const double local_steering = localLeftSteering(right_clearance);
                const double local_target = away_direction * local_steering;
                if (local_hard_left_turn_ ||
                    filtered_steering_ * away_direction < local_steering)
                {
                    filtered_steering_ = local_target;
                }
            }

            geometry_msgs::msg::Twist command;
            const double turn_ratio = max_steering_ > 0.0 ?
                std::abs(filtered_steering_) / max_steering_ : 0.0;
            if (local_left_turn_active_) {
                command.linear.x = localTurnSpeed(right_clearance);
            } else if (followed_cone_in_front || right_turn_approach_active_) {
                command.linear.x = min_drive_speed_;
            } else {
                command.linear.x = std::max(
                    min_drive_speed_, base_speed_ * (1.0 - 0.10 * turn_ratio));
            }
            command.angular.z = filtered_steering_;
            velocity_pub_->publish(command);

            RCLCPP_INFO_THROTTLE(
                get_logger(), *get_clock(), 500,
                "跟随%s侧：锥桶=%zu, 拟合内点=%zu, 侧距=%.2fm(原始%.2f), "
                "距离状态=%s, 边界方向=%.1fdeg, 模式=%s, 弯道基础=%.3f, "
                "局部距离[右=%.2f,右前=%.2f,前=%.2f], 右侧最近=%.2f, "
                "右弯接近=%s, 重入冷却=%s, "
                "局部左转=%s, 左弯峰值=%s, 出口确认=%d, "
                "安全避让=%s, 前方绕行=%s, 速度=%.2f(PWM %.0f), "
                "转向=%.3f(舵机PWM %.0f)",
                following_left ? "左" : "右", active_cones.size(),
                raw_estimate.inlier_count, filtered_clearance_, raw_estimate.clearance,
                filtered_clearance_ < desired_clearance_ - clearance_deadband_ ? "过近" :
                    (filtered_clearance_ > desired_clearance_ + clearance_deadband_ ? "过远" : "合适"),
                filtered_boundary_heading_ * 180.0 / kPi, curve_mode_ ? "大弯" : "普通",
                curve_base_steering_,
                local_distances.right, local_distances.right_front, local_distances.front,
                right_clearance,
                right_turn_approach_active_ ? "是" : "否",
                right_turn_reentry_blocked_ ? "是" : "否",
                local_left_turn_active_ ?
                    (clearance_escape_active_ ? "侧距渐进" :
                     (local_hard_left_turn_ ? "强" :
                      (closeClearanceRatio(right_clearance) > 0.0 ? "渐进" : "普通"))) :
                    "否",
                local_left_peak_seen_ ? "是" : "否",
                left_exit_confirm_count_,
                clearance_safety_active_ ? "是" : "否",
                front_avoidance_active ? "是" : "否",
                command.linear.x, 1500.0 + command.linear.x * 100.0,
                command.angular.z, 1500.0 - command.angular.z * 1300.0);
            return;
        }

        const double lost_seconds = has_seen_target_ ? (now - last_target_time_).seconds() :
            std::numeric_limits<double>::infinity();
        if (local_left_turn_active_) {
            const bool escaping_followed_side =
                canEscapeFollowedSideObstacle(front_obstacle, following_left);
            if (front_obstacle.found && frontObstacleIsEmergency(front_obstacle) &&
                !escaping_followed_side)
            {
                publishStop("局部左转时正前方存在碰撞风险，停车");
                return;
            }

            const double local_steering = localLeftSteering(right_clearance);
            filtered_steering_ = away_direction * local_steering;

            geometry_msgs::msg::Twist command;
            command.linear.x = localTurnSpeed(right_clearance);
            command.angular.z = filtered_steering_;
            velocity_pub_->publish(command);
            RCLCPP_WARN_THROTTLE(
                get_logger(), *get_clock(), 500,
                "右边界暂时丢失，依据局部距离继续%s左转："
                "右=%.2f, 右前=%.2f, 前=%.2f, 最近右侧=%.2f, 转向=%.3f",
                clearance_escape_active_ ? "侧距渐进" :
                    (local_hard_left_turn_ ? "强" :
                     (closeClearanceRatio(right_clearance) > 0.0 ? "渐进" : "普通")),
                local_distances.right, local_distances.right_front,
                local_distances.front, right_clearance, command.angular.z);
            return;
        }
        if (front_avoidance_active) {
            if (front_obstacle.found &&
                front_obstacle.range <= front_emergency_distance_)
            {
                publishStop("丢失边界且正前方存在碰撞风险，停车");
                return;
            }

            const double avoidance_target =
                away_direction * front_avoidance_steering_;
            filtered_steering_ =
                0.80 * avoidance_target + 0.20 * filtered_steering_;

            geometry_msgs::msg::Twist command;
            command.linear.x = min_drive_speed_;
            command.angular.z = filtered_steering_;
            velocity_pub_->publish(command);
            RCLCPP_WARN_THROTTLE(
                get_logger(), *get_clock(), 500,
                "边界短暂丢失，保持前方绕行转向：转向=%.3f",
                command.angular.z);
            return;
        }

        if (front_obstacle.found &&
            (frontObstacleIsEmergency(front_obstacle) ||
             front_obstacle.range <= front_stop_distance_))
        {
            publishStop("丢失边界且正前方存在碰撞风险，停车");
            return;
        }

        if (lost_seconds <= target_loss_timeout_) {
            geometry_msgs::msg::Twist command;
            command.linear.x = search_speed_;
            command.angular.z = filtered_steering_;
            velocity_pub_->publish(command);
            RCLCPP_WARN_THROTTLE(
                get_logger(), *get_clock(), 500,
                "%s侧锥桶暂时丢失 %.2fs，低速保持上一转向",
                following_left ? "左" : "右", lost_seconds);
        } else {
            has_boundary_state_ = false;
            curve_mode_ = false;
            curve_direction_ = 0.0;
            curve_base_steering_ = 0.0;
            curve_exit_count_ = 0;
            curve_reverse_count_ = 0;
            resetRightTurnControl();
            publishStop("单侧锥桶持续丢失，停车");
        }
    }

    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr velocity_pub_;

    std::string follow_side_;
    std::string active_side_;
    double desired_clearance_{};
    double lookahead_distance_{};
    double min_detection_range_{};
    double max_detection_range_{};
    double side_sector_angle_{};
    double front_dead_zone_{};
    double cluster_gap_{};
    double min_cone_width_{};
    double max_cone_width_{};
    int min_cluster_points_{};
    double base_speed_{};
    double search_speed_{};
    double min_drive_speed_{};
    double lateral_gain_{};
    double heading_gain_{};
    double clearance_deadband_{};
    double curve_enter_heading_{};
    double curve_exit_heading_{};
    int curve_exit_frames_{};
    double curve_reverse_heading_{};
    int curve_reverse_frames_{};
    double curve_min_steering_{};
    double curve_steering_update_{};
    double boundary_inlier_tolerance_{};
    double boundary_smoothing_{};
    double max_boundary_distance_jump_{};
    double max_boundary_heading_jump_{};
    double steering_sign_{};
    double max_steering_{};
    double max_toward_boundary_steering_{};
    double min_away_curve_steering_{};
    double minimum_safe_clearance_{};
    double steering_smoothing_{};
    double target_loss_timeout_{};
    double front_stop_distance_{};
    double front_stop_angle_{};
    double front_avoidance_distance_{};
    double front_avoidance_center_angle_{};
    double front_avoidance_steering_{};
    double front_avoidance_hold_time_{};
    double front_avoidance_max_duration_{};
    double front_emergency_distance_{};
    double front_emergency_angle_{};
    double followed_cone_match_distance_{};
    double local_turn_distance_{};
    double local_turn_release_distance_{};
    int local_turn_confirm_frames_{};
    int local_turn_release_frames_{};
    double local_front_sector_angle_{};
    double local_right_front_sector_angle_{};
    double right_turn_far_margin_{};
    double right_turn_preview_distance_{};
    double right_turn_preview_release_distance_{};
    double right_turn_detect_steering_{};
    double critical_turn_speed_{};
    double front_escape_stop_distance_{};
    double left_exit_peak_heading_{};
    double left_exit_heading_{};
    int left_exit_confirm_frames_{};
    double right_turn_reentry_cooldown_{};

    bool has_seen_target_{false};
    bool has_boundary_state_{false};
    bool curve_mode_{false};
    bool clearance_safety_active_{false};
    bool front_avoidance_mode_{false};
    bool right_turn_approach_active_{false};
    bool local_left_turn_active_{false};
    bool local_hard_left_turn_{false};
    bool clearance_escape_active_{false};
    bool local_left_peak_seen_{false};
    bool has_left_exit_time_{false};
    bool right_turn_reentry_blocked_{false};
    int curve_exit_count_{0};
    int curve_reverse_count_{0};
    int local_turn_confirm_count_{0};
    int local_turn_release_count_{0};
    int left_exit_confirm_count_{0};
    rclcpp::Time last_target_time_{0, 0, RCL_ROS_TIME};
    rclcpp::Time front_avoidance_started_time_{0, 0, RCL_ROS_TIME};
    rclcpp::Time front_avoidance_last_seen_time_{0, 0, RCL_ROS_TIME};
    rclcpp::Time last_left_exit_time_{0, 0, RCL_ROS_TIME};
    double filtered_clearance_{0.0};
    double filtered_boundary_heading_{0.0};
    double filtered_steering_{0.0};
    double curve_direction_{0.0};
    double curve_base_steering_{0.0};
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LidarProcessor>());
    rclcpp::shutdown();
    return 0;
}
