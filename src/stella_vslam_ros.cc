#include <stella_vslam_ros.h>
#include <stella_vslam/publish/map_publisher.h>
#include <stella_vslam/data/keyframe.h>
#include <stella_vslam/data/landmark.h>
#include <stella_vslam/data/graph_node.h>
#include <stella_vslam/camera/base.h>
#include <stella_vslam/camera/perspective.h>
#include <stella_vslam/camera/fisheye.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <chrono>

#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <geometry_msgs/msg/transform_stamped.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <Eigen/Geometry>

namespace {
Eigen::Affine3d project_to_xy_plane(const Eigen::Affine3d& affine) {
    Eigen::Matrix4d mat = affine.matrix();
    mat(2, 3) = 0.0;
    Eigen::Translation<double, 3> trans(mat.col(3).head<3>());
    double rx = mat(0, 0);
    double ry = mat(1, 0);
    double yaw = std::atan2(ry, rx);
    return trans * Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ());
}
} // namespace

namespace stella_vslam_ros {
system::system(const std::shared_ptr<stella_vslam::system>& slam,
               rclcpp::Node* node,
               const std::string& mask_img_path)
    : slam_(slam), node_(node), custom_qos_(rmw_qos_profile_sensor_data),
      mask_(mask_img_path.empty() ? cv::Mat{} : cv::imread(mask_img_path, cv::IMREAD_GRAYSCALE)),
      pose_pub_(node_->create_publisher<nav_msgs::msg::Odometry>("~/camera_pose", 1)),
      keyframes_pub_(node_->create_publisher<geometry_msgs::msg::PoseArray>("~/keyframes", 1)),
      keyframes_2d_pub_(node_->create_publisher<geometry_msgs::msg::PoseArray>("~/keyframes_2d", 1)),
      map_to_odom_broadcaster_(std::make_shared<tf2_ros::TransformBroadcaster>(node_)),
      tf_(std::make_unique<tf2_ros::Buffer>(node_->get_clock())),
      transform_listener_(std::make_shared<tf2_ros::TransformListener>(*tf_)) {
    custom_qos_.depth = 1;
    init_pose_sub_ = node_->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
        "/initialpose", 1,
        std::bind(&system::init_pose_callback,
                  this, std::placeholders::_1));
    setParams();
    rot_ros_to_cv_map_frame_ = (Eigen::Matrix3d() << 0, 0, 1,
                                -1, 0, 0,
                                0, -1, 0)
                                   .finished();
    // Pose-graph publisher on the OKVIS2-X contract topic. QoS matches OKVIS
    // (KeepLast(2), reliable) so a downstream z-floc PGO / rosbag recorder sees it.
    pose_graph_pub_ = node_->create_publisher<okvis_pose_graph_msgs::msg::PoseGraph>(
        pose_graph_topic_, rclcpp::QoS(rclcpp::KeepLast(2)));
}

void system::publish_pose(const Eigen::Matrix4d& cam_pose_wc, const rclcpp::Time& stamp) {
    // Extract rotation matrix and translation vector from
    Eigen::Matrix3d rot(cam_pose_wc.block<3, 3>(0, 0));
    Eigen::Translation3d trans(cam_pose_wc.block<3, 1>(0, 3));
    Eigen::Affine3d map_to_camera_affine(trans * rot);

    // Transform map frame from CV coordinate system to ROS coordinate system
    map_to_camera_affine.prerotate(rot_ros_to_cv_map_frame_);

    // Create odometry message and update it with current camera pose
    nav_msgs::msg::Odometry pose_msg;
    pose_msg.header.stamp = stamp;
    pose_msg.header.frame_id = map_frame_;
    pose_msg.child_frame_id = camera_frame_;
    pose_msg.pose.pose = tf2::toMsg(map_to_camera_affine * rot_ros_to_cv_map_frame_.inverse());
    pose_pub_->publish(pose_msg);

    // Send map->odom transform. Set publish_tf to false if not using TF
    if (publish_tf_) {
        try {
            auto camera_to_odom = tf_->lookupTransform(
                camera_optical_frame_, odom_frame_, tf2_ros::fromMsg(builtin_interfaces::msg::Time(stamp)),
                tf2::durationFromSec(0.0));
            Eigen::Affine3d camera_to_odom_affine = tf2::transformToEigen(camera_to_odom.transform);

            geometry_msgs::msg::TransformStamped map_to_odom_msg;
            if (odom2d_) {
                Eigen::Affine3d map_to_camera_affine_2d = project_to_xy_plane(map_to_camera_affine * rot_ros_to_cv_map_frame_.inverse()) * rot_ros_to_cv_map_frame_;
                Eigen::Affine3d camera_to_odom_affine_2d = (project_to_xy_plane(camera_to_odom_affine.inverse() * rot_ros_to_cv_map_frame_.inverse()) * rot_ros_to_cv_map_frame_).inverse();
                Eigen::Affine3d map_to_odom_affine_2d = map_to_camera_affine_2d * camera_to_odom_affine_2d;
                map_to_odom_msg = tf2::eigenToTransform(map_to_odom_affine_2d);
            }
            else {
                map_to_odom_msg = tf2::eigenToTransform(map_to_camera_affine * camera_to_odom_affine);
            }
            tf2::TimePoint transform_timestamp = tf2_ros::fromMsg(stamp) + tf2::durationFromSec(transform_tolerance_);
            map_to_odom_msg.header.stamp = tf2_ros::toMsg(transform_timestamp);
            map_to_odom_msg.header.frame_id = map_frame_;
            map_to_odom_msg.child_frame_id = odom_frame_;
            map_to_odom_broadcaster_->sendTransform(map_to_odom_msg);
        }
        catch (tf2::TransformException& ex) {
            RCLCPP_ERROR_STREAM_THROTTLE(node_->get_logger(), *node_->get_clock(), 1000, "Transform failed: " << ex.what());
        }
    }
}

void system::publish_keyframes(const rclcpp::Time& stamp) {
    geometry_msgs::msg::PoseArray keyframes_msg;
    geometry_msgs::msg::PoseArray keyframes_2d_msg;
    keyframes_msg.header.stamp = stamp;
    keyframes_msg.header.frame_id = map_frame_;
    keyframes_2d_msg.header = keyframes_msg.header;
    std::vector<std::shared_ptr<stella_vslam::data::keyframe>> all_keyfrms;
    slam_->get_map_publisher()->get_keyframes(all_keyfrms);
    for (const auto& keyfrm : all_keyfrms) {
        if (!keyfrm || keyfrm->will_be_erased()) {
            continue;
        }
        Eigen::Matrix4d cam_pose_wc = keyfrm->get_pose_wc();
        Eigen::Matrix3d rot(cam_pose_wc.block<3, 3>(0, 0));
        Eigen::Translation3d trans(cam_pose_wc.block<3, 1>(0, 3));
        Eigen::Affine3d map_to_camera_affine(trans * rot);
        Eigen::Affine3d pose_affine = rot_ros_to_cv_map_frame_ * map_to_camera_affine * rot_ros_to_cv_map_frame_.inverse();
        keyframes_msg.poses.push_back(tf2::toMsg(pose_affine));
        keyframes_2d_msg.poses.push_back(tf2::toMsg(project_to_xy_plane(pose_affine)));
    }
    keyframes_pub_->publish(keyframes_msg);
    keyframes_2d_pub_->publish(keyframes_2d_msg);
}

namespace {

// Pinhole intrinsics + image bounds pulled from a stella camera model. Fisheye
// keyframes store undistorted keypoints, so a pinhole reprojection Jacobian with
// the model fx/fy is the right first-order information (same convention OV-SLAM /
// DROID-W use for their actual-Hessian edge strength). Equirectangular has no
// single focal -> not supported (caller falls back to a covisibility proxy).
struct pinhole_intrinsics {
    bool ok = false;
    double fx = 0, fy = 0, cx = 0, cy = 0;
    double min_x = 0, max_x = 0, min_y = 0, max_y = 0;
};

pinhole_intrinsics get_pinhole(const stella_vslam::camera::base* cam) {
    pinhole_intrinsics k;
    if (!cam) return k;
    if (cam->model_type_ == stella_vslam::camera::model_type_t::Perspective) {
        const auto* c = static_cast<const stella_vslam::camera::perspective*>(cam);
        k.fx = c->fx_; k.fy = c->fy_; k.cx = c->cx_; k.cy = c->cy_;
    }
    else if (cam->model_type_ == stella_vslam::camera::model_type_t::Fisheye) {
        const auto* c = static_cast<const stella_vslam::camera::fisheye*>(cam);
        k.fx = c->fx_; k.fy = c->fy_; k.cx = c->cx_; k.cy = c->cy_;
    }
    else {
        return k; // unsupported model
    }
    k.min_x = cam->img_bounds_.min_x_; k.max_x = cam->img_bounds_.max_x_;
    k.min_y = cam->img_bounds_.min_y_; k.max_y = cam->img_bounds_.max_y_;
    k.ok = true;
    return k;
}

inline Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
    Eigen::Matrix3d S;
    S << 0, -v.z(), v.y(),
        v.z(), 0, -v.x(),
        -v.y(), v.x(), 0;
    return S;
}

// Accumulate the 2x6 reprojection Jacobian outer product (w * J^T J) of one
// world landmark observed by keyframe `kf` into JtJ. se3 tangent order is
// [translation(0..2), rotation(3..5)] (left-perturbation), matching OV-SLAM /
// DROID-W. Returns false if the landmark is behind / too far / outside the image.
bool accumulate_pose_hessian(const std::shared_ptr<stella_vslam::data::keyframe>& kf,
                             const pinhole_intrinsics& k,
                             const Eigen::Vector3d& pos_w,
                             double w,
                             Eigen::Matrix<double, 6, 6>& JtJ) {
    const Eigen::Matrix3d R_cw = kf->get_rot_cw();
    const Eigen::Vector3d t_cw = kf->get_trans_cw();
    const Eigen::Vector3d pc = R_cw * pos_w + t_cw;
    const double z = pc.z();
    if (z < 0.1 || z > 80.0) return false;
    const double iz = 1.0 / z;
    const double u = k.fx * pc.x() * iz + k.cx;
    const double v = k.fy * pc.y() * iz + k.cy;
    if (u < k.min_x || u > k.max_x || v < k.min_y || v > k.max_y) return false;

    Eigen::Matrix<double, 2, 3> Jp;
    Jp << k.fx * iz, 0.0, -k.fx * pc.x() * iz * iz,
        0.0, k.fy * iz, -k.fy * pc.y() * iz * iz;
    Eigen::Matrix<double, 2, 6> J;
    J.leftCols<3>() = Jp;                // d(reproj)/d(translation)
    J.rightCols<3>() = -Jp * skew(pc);   // d(reproj)/d(rotation)
    JtJ.noalias() += w * J.transpose() * J;
    return true;
}

// Actual reprojection information for the relative pose between two keyframes,
// summed over their shared landmarks (bidirectional average of each frame's
// Hessian), reduced to scalar trans/rot info = geomean of the diagonal triples.
// Returns false (caller uses covisibility proxy) when fewer than 8 usable shared
// landmarks or an unsupported camera model.
bool compute_edge_hessian_info(const std::shared_ptr<stella_vslam::data::keyframe>& kf_a,
                               const std::shared_ptr<stella_vslam::data::keyframe>& kf_b,
                               double sigma_px,
                               double& info_trans, double& info_rot) {
    const pinhole_intrinsics ka = get_pinhole(kf_a->camera_);
    const pinhole_intrinsics kb = get_pinhole(kf_b->camera_);
    if (!ka.ok || !kb.ok) return false;

    const double w = 1.0 / (sigma_px * sigma_px);
    Eigen::Matrix<double, 6, 6> JtJ_a = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 6> JtJ_b = Eigen::Matrix<double, 6, 6>::Zero();
    int n_used = 0;

    const auto lms_a = kf_a->get_landmarks();
    for (const auto& lm : lms_a) {
        if (!lm || lm->will_be_erased()) continue;
        if (lm->get_index_in_keyframe(kf_b) < 0) continue; // not shared with B
        const Eigen::Vector3d pos_w = lm->get_pos_in_world();
        const bool oa = accumulate_pose_hessian(kf_a, ka, pos_w, w, JtJ_a);
        const bool ob = accumulate_pose_hessian(kf_b, kb, pos_w, w, JtJ_b);
        if (oa && ob) ++n_used;
    }
    if (n_used < 8) return false;

    const Eigen::Matrix<double, 6, 1> diag =
        (0.5 * (JtJ_a + JtJ_b)).diagonal().cwiseMax(1e-6);
    info_trans = std::exp(diag.head<3>().array().log().mean());
    info_rot = std::exp(diag.tail<3>().array().log().mean());
    return true;
}

} // namespace

void system::publish_pose_graph(const rclcpp::Time& stamp) {
    using stella_vslam::data::keyframe;

    // Throttle: the snapshot rebuild is O(edges) and runs in the tracking callback,
    // so cap its rate to avoid starving real-time tracking (OKVIS-style).
    const double now_sec = stamp.seconds();
    if (last_pose_graph_pub_sec_ >= 0.0
        && (now_sec - last_pose_graph_pub_sec_) < pose_graph_min_interval_) {
        return;
    }

    std::vector<std::shared_ptr<keyframe>> raw_kfs;
    slam_->get_map_publisher()->get_keyframes(raw_kfs);

    // Keep only live keyframes, sorted ascending by id (= vertex id).
    std::vector<std::shared_ptr<keyframe>> kfs;
    kfs.reserve(raw_kfs.size());
    for (const auto& kf : raw_kfs) {
        if (kf && !kf->will_be_erased()) kfs.push_back(kf);
    }
    if (kfs.size() < 2) return;
    // Incremental (growing) publication: only emit when the keyframe set changed
    // (a new KF was inserted or an old one culled) — mirrors OKVIS2-X per-KF grow.
    if (kfs.size() == last_pose_graph_num_kfs_) return;
    last_pose_graph_num_kfs_ = kfs.size();
    std::sort(kfs.begin(), kfs.end(),
              [](const std::shared_ptr<keyframe>& a, const std::shared_ptr<keyframe>& b) {
                  return a->id_ < b->id_;
              });

    okvis_pose_graph_msgs::msg::PoseGraph msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = map_frame_;

    auto to_pose = [](const Eigen::Matrix4d& T) {
        geometry_msgs::msg::Pose p;
        const Eigen::Vector3d r = T.block<3, 1>(0, 3);
        const Eigen::Quaterniond q(Eigen::Matrix3d(T.block<3, 3>(0, 0)));
        p.position.x = r.x(); p.position.y = r.y(); p.position.z = r.z();
        p.orientation.x = q.x(); p.orientation.y = q.y();
        p.orientation.z = q.z(); p.orientation.w = q.w();
        return p;
    };

    // --- Vertices: T_WS = camera->world (native stella / CV frame, up-to-scale
    //     for monocular). Self-consistent with edge_rel below. ---
    std::map<unsigned int, std::shared_ptr<keyframe>> kf_by_id;
    for (const auto& kf : kfs) {
        kf_by_id.emplace(kf->id_, kf);
        msg.vertex_id.push_back(static_cast<uint64_t>(kf->id_));
        msg.vertex_stamp_ns.push_back(static_cast<int64_t>(kf->timestamp_ * 1e9));
        msg.vertex_pose.push_back(to_pose(kf->get_pose_wc()));
    }

    // --- Edges. Dedup on (i<j); prefer VO(type0) for consecutive KFs, else
    //     covis/loop(type1). Strength = actual reprojection Hessian info. ---
    struct edge_info { uint8_t type; int covis; };
    std::map<std::pair<unsigned int, unsigned int>, edge_info> edges;

    // Sequential VO edges (type 0) between consecutive keyframes by id.
    for (size_t i = 1; i < kfs.size(); ++i) {
        const unsigned int a = kfs[i - 1]->id_, b = kfs[i]->id_;
        const int cov = static_cast<int>(kfs[i]->graph_node_->get_num_shared_landmarks(kfs[i - 1]));
        edges[{a, b}] = edge_info{0, cov};
    }
    // Covisibility + loop edges (type 1).
    for (const auto& kf : kfs) {
        const auto loop_edges = kf->graph_node_->get_loop_edges();
        const auto covis = kf->graph_node_->get_covisibilities();
        for (const auto& nb : covis) {
            if (!nb || nb->will_be_erased()) continue;
            if (!kf_by_id.count(nb->id_)) continue;
            const int cov = static_cast<int>(kf->graph_node_->get_num_shared_landmarks(nb));
            const bool is_loop = loop_edges.count(nb) > 0;
            if (!is_loop && cov < pose_graph_min_covisibility_) continue;
            const unsigned int a = std::min(kf->id_, nb->id_);
            const unsigned int b = std::max(kf->id_, nb->id_);
            if (a == b) continue;
            auto it = edges.find({a, b});
            if (it == edges.end()) {
                edges[{a, b}] = edge_info{1, cov};
            }
            else if (it->second.type == 0) {
                // keep VO type but remember covisibility count / loop flag
                it->second.covis = std::max(it->second.covis, cov);
                if (is_loop) it->second.type = 1;
            }
        }
    }

    for (const auto& kv : edges) {
        const unsigned int ia = kv.first.first, ib = kv.first.second;
        const auto& kf_a = kf_by_id.at(ia);
        const auto& kf_b = kf_by_id.at(ib);
        // edge_rel = T_AB = T_WA^-1 * T_WB = pose_cw(A) * pose_wc(B)
        const Eigen::Matrix4d T_AB = kf_a->get_pose_cw() * kf_b->get_pose_wc();

        double info_trans = 0.0, info_rot = 0.0;
        if (!compute_edge_hessian_info(kf_a, kf_b, pose_graph_sigma_px_, info_trans, info_rot)) {
            // Fallback: covisibility proxy (still per-edge, never a constant).
            const double proxy = 500.0 * std::max(1, kv.second.covis);
            info_trans = proxy;
            info_rot = proxy;
        }

        msg.edge_i.push_back(static_cast<uint64_t>(ia));
        msg.edge_j.push_back(static_cast<uint64_t>(ib));
        msg.edge_rel.push_back(to_pose(T_AB));
        msg.edge_info_trans.push_back(info_trans);
        msg.edge_info_rot.push_back(info_rot);
        msg.edge_type.push_back(kv.second.type);
    }

    pose_graph_pub_->publish(msg);
    last_pose_graph_pub_sec_ = now_sec;
}

void system::setParams() {
    odom_frame_ = std::string("odom");
    odom_frame_ = node_->declare_parameter("odom_frame", odom_frame_);

    map_frame_ = std::string("map");
    map_frame_ = node_->declare_parameter("map_frame", map_frame_);

    robot_base_frame_ = std::string("base_link");
    robot_base_frame_ = node_->declare_parameter("robot_base_frame", robot_base_frame_);

    camera_frame_ = std::string("camera_frame");
    camera_frame_ = node_->declare_parameter("camera_frame", camera_frame_);

    publish_tf_ = true;
    publish_tf_ = node_->declare_parameter("publish_tf", publish_tf_);

    odom2d_ = false;
    odom2d_ = node_->declare_parameter("odom2d", odom2d_);

    publish_keyframes_ = true;
    publish_keyframes_ = node_->declare_parameter("publish_keyframes", publish_keyframes_);

    publish_pose_graph_ = true;
    publish_pose_graph_ = node_->declare_parameter("publish_pose_graph", publish_pose_graph_);

    pose_graph_topic_ = std::string("/okvis/okvis_pose_graph");
    pose_graph_topic_ = node_->declare_parameter("pose_graph_topic", pose_graph_topic_);

    pose_graph_min_covisibility_ = 15;
    pose_graph_min_covisibility_ = node_->declare_parameter("pose_graph_min_covisibility", pose_graph_min_covisibility_);

    pose_graph_sigma_px_ = 1.0;
    pose_graph_sigma_px_ = node_->declare_parameter("pose_graph_sigma_px", pose_graph_sigma_px_);

    pose_graph_min_interval_ = 1.0;
    pose_graph_min_interval_ = node_->declare_parameter("pose_graph_min_interval", pose_graph_min_interval_);

    transform_tolerance_ = 0.5;
    transform_tolerance_ = node_->declare_parameter("transform_tolerance", transform_tolerance_);

    encoding_ = "";
    encoding_ = node_->declare_parameter("encoding", encoding_);
}

void system::init_pose_callback(
    const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
    if (camera_optical_frame_.empty()) {
        RCLCPP_ERROR(node_->get_logger(),
                     "Camera link is not set: no images were received yet");
        return;
    }

    Eigen::Translation3d trans(
        msg->pose.pose.position.x,
        msg->pose.pose.position.y,
        msg->pose.pose.position.z);
    Eigen::Quaterniond rot_q(
        msg->pose.pose.orientation.w,
        msg->pose.pose.orientation.x,
        msg->pose.pose.orientation.y,
        msg->pose.pose.orientation.z);
    Eigen::Affine3d initialpose_affine(trans * rot_q);

    Eigen::Matrix3d rot_cv_to_ros_map_frame;
    rot_cv_to_ros_map_frame << 0, -1, 0,
        0, 0, -1,
        1, 0, 0;

    Eigen::Affine3d map_to_initialpose_frame_affine;
    try {
        auto map_to_initialpose_frame = tf_->lookupTransform(
            map_frame_, msg->header.frame_id, tf2_ros::fromMsg(msg->header.stamp),
            tf2::durationFromSec(0.0));
        map_to_initialpose_frame_affine = tf2::transformToEigen(
            map_to_initialpose_frame.transform);
    }
    catch (tf2::TransformException& ex) {
        RCLCPP_ERROR_STREAM_THROTTLE(node_->get_logger(), *node_->get_clock(), 1000, "Transform failed: " << ex.what());
        return;
    }

    Eigen::Affine3d robot_base_frame_to_camera_affine;
    try {
        auto robot_base_frame_to_camera = tf_->lookupTransform(
            robot_base_frame_, camera_optical_frame_, tf2_ros::fromMsg(msg->header.stamp),
            tf2::durationFromSec(0.0));
        robot_base_frame_to_camera_affine = tf2::transformToEigen(robot_base_frame_to_camera.transform);
    }
    catch (tf2::TransformException& ex) {
        RCLCPP_ERROR_STREAM_THROTTLE(node_->get_logger(), *node_->get_clock(), 1000, "Transform failed: " << ex.what());
        return;
    }

    // Target transform is map_cv -> camera_link and known parameters are following:
    //   rot_cv_to_ros_map_frame: T(map_cv -> map)
    //   map_to_initialpose_frame_affine: T(map -> `msg->header.frame_id`)
    //   initialpose_affine: T(`msg->header.frame_id` -> base_link)
    //   robot_base_frame_to_camera_affine: T(base_link -> camera_link)
    // The flow of the transformation is as follows:
    //   map_cv -> map -> `msg->header.frame_id` -> base_link -> camera_link
    Eigen::Matrix4d cam_pose_cv = (rot_cv_to_ros_map_frame * map_to_initialpose_frame_affine
                                   * initialpose_affine * robot_base_frame_to_camera_affine)
                                      .matrix();

    const Eigen::Vector3d normal_vector = (Eigen::Vector3d() << 0., 1., 0.).finished();
    if (!slam_->relocalize_by_pose_2d(cam_pose_cv, normal_vector)) {
        RCLCPP_ERROR(node_->get_logger(), "Can not set initial pose");
    }
}

mono::mono(const std::shared_ptr<stella_vslam::system>& slam,
           rclcpp::Node* node,
           const std::string& mask_img_path)
    : system(slam, node, mask_img_path) {
    auto qos = rclcpp::QoS(rclcpp::QoSInitialization::from_rmw(custom_qos_), custom_qos_);
    raw_image_sub_ = node_->create_subscription<sensor_msgs::msg::Image>(
        "camera/image_raw", qos, [this](sensor_msgs::msg::Image::UniquePtr msg_unique_ptr) { callback(std::move(msg_unique_ptr)); });
}
void mono::callback(sensor_msgs::msg::Image::UniquePtr msg_unique_ptr) {
    sensor_msgs::msg::Image::ConstSharedPtr msg = std::move(msg_unique_ptr);
    if (camera_optical_frame_.empty()) {
        camera_optical_frame_ = msg->header.frame_id;
    }
    const rclcpp::Time tp_1 = node_->now();
    const double timestamp = rclcpp::Time(msg->header.stamp).seconds();

    // input the current frame and estimate the camera pose
    auto cam_pose_wc = slam_->feed_monocular_frame(cv_bridge::toCvShare(msg, encoding_)->image, timestamp, mask_);

    const rclcpp::Time tp_2 = node_->now();
    const double track_time = (tp_2 - tp_1).seconds();

    // track times in seconds
    track_times_.push_back(track_time);

    if (cam_pose_wc) {
        publish_pose(*cam_pose_wc, msg->header.stamp);
    }
    if (publish_keyframes_) {
        publish_keyframes(msg->header.stamp);
    }
    if (publish_pose_graph_) {
        publish_pose_graph(msg->header.stamp);
    }
}

void mono::callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
    if (camera_optical_frame_.empty()) {
        camera_optical_frame_ = msg->header.frame_id;
    }
    const rclcpp::Time tp_1 = node_->now();
    const double timestamp = rclcpp::Time(msg->header.stamp).seconds();

    // input the current frame and estimate the camera pose
    auto cam_pose_wc = slam_->feed_monocular_frame(cv_bridge::toCvShare(msg, encoding_)->image, timestamp, mask_);

    const rclcpp::Time tp_2 = node_->now();
    const double track_time = (tp_2 - tp_1).seconds();

    // track times in seconds
    track_times_.push_back(track_time);

    if (cam_pose_wc) {
        publish_pose(*cam_pose_wc, msg->header.stamp);
    }
    if (publish_keyframes_) {
        publish_keyframes(msg->header.stamp);
    }
    if (publish_pose_graph_) {
        publish_pose_graph(msg->header.stamp);
    }
}

stereo::stereo(const std::shared_ptr<stella_vslam::system>& slam,
               rclcpp::Node* node,
               const std::string& mask_img_path,
               const std::shared_ptr<stella_vslam::util::stereo_rectifier>& rectifier)
    : system(slam, node, mask_img_path),
      rectifier_(rectifier),
      left_sf_(node_, "camera/left/image_raw"),
      right_sf_(node_, "camera/right/image_raw") {
    use_exact_time_ = false;
    use_exact_time_ = node_->declare_parameter("use_exact_time", use_exact_time_);
    if (use_exact_time_) {
        exact_time_sync_ = std::make_shared<ExactTimeSyncPolicy::Sync>(2, left_sf_, right_sf_);
        exact_time_sync_->registerCallback(&stereo::callback, this);
    }
    else {
        approx_time_sync_ = std::make_shared<ApproximateTimeSyncPolicy::Sync>(10, left_sf_, right_sf_);
        approx_time_sync_->registerCallback(&stereo::callback, this);
    }
}

void stereo::callback(const sensor_msgs::msg::Image::ConstSharedPtr& left, const sensor_msgs::msg::Image::ConstSharedPtr& right) {
    if (camera_optical_frame_.empty()) {
        camera_optical_frame_ = left->header.frame_id;
    }
    auto leftcv = cv_bridge::toCvShare(left, encoding_)->image;
    auto rightcv = cv_bridge::toCvShare(right, encoding_)->image;
    if (leftcv.empty() || rightcv.empty()) {
        return;
    }

    if (rectifier_) {
        rectifier_->rectify(leftcv, rightcv, leftcv, rightcv);
    }

    const rclcpp::Time tp_1 = node_->now();
    const double timestamp = rclcpp::Time(left->header.stamp).seconds();

    // input the current frame and estimate the camera pose
    auto cam_pose_wc = slam_->feed_stereo_frame(leftcv, rightcv, timestamp, mask_);

    const rclcpp::Time tp_2 = node_->now();
    const double track_time = (tp_2 - tp_1).seconds();

    // track times in seconds
    track_times_.push_back(track_time);

    if (cam_pose_wc) {
        publish_pose(*cam_pose_wc, left->header.stamp);
    }
    if (publish_keyframes_) {
        publish_keyframes(left->header.stamp);
    }
    if (publish_pose_graph_) {
        publish_pose_graph(left->header.stamp);
    }
}

rgbd::rgbd(const std::shared_ptr<stella_vslam::system>& slam,
           rclcpp::Node* node,
           const std::string& mask_img_path)
    : system(slam, node, mask_img_path),
      color_sf_(node_, "camera/color/image_raw"),
      depth_sf_(node_, "camera/depth/image_raw") {
    use_exact_time_ = false;
    use_exact_time_ = node_->declare_parameter("use_exact_time", use_exact_time_);
    if (use_exact_time_) {
        exact_time_sync_ = std::make_shared<ExactTimeSyncPolicy::Sync>(2, color_sf_, depth_sf_);
        exact_time_sync_->registerCallback(&rgbd::callback, this);
    }
    else {
        approx_time_sync_ = std::make_shared<ApproximateTimeSyncPolicy::Sync>(10, color_sf_, depth_sf_);
        approx_time_sync_->registerCallback(&rgbd::callback, this);
    }
}

void rgbd::callback(const sensor_msgs::msg::Image::ConstSharedPtr& color, const sensor_msgs::msg::Image::ConstSharedPtr& depth) {
    if (camera_optical_frame_.empty()) {
        camera_optical_frame_ = color->header.frame_id;
    }
    auto colorcv = cv_bridge::toCvShare(color, encoding_)->image;
    auto depthcv = cv_bridge::toCvShare(depth)->image;
    if (colorcv.empty() || depthcv.empty()) {
        return;
    }
    if (depthcv.type() == CV_32FC1) {
        cv::patchNaNs(depthcv);
    }

    const rclcpp::Time tp_1 = node_->now();
    const double timestamp = rclcpp::Time(color->header.stamp).seconds();

    // input the current frame and estimate the camera pose
    auto cam_pose_wc = slam_->feed_RGBD_frame(colorcv, depthcv, timestamp, mask_);

    const rclcpp::Time tp_2 = node_->now();
    const double track_time = (tp_2 - tp_1).seconds();

    // track time in seconds
    track_times_.push_back(track_time);

    if (cam_pose_wc) {
        publish_pose(*cam_pose_wc, color->header.stamp);
    }
    if (publish_keyframes_) {
        publish_keyframes(color->header.stamp);
    }
    if (publish_pose_graph_) {
        publish_pose_graph(color->header.stamp);
    }
}

} // namespace stella_vslam_ros
