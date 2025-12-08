#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <lo_sam/msg/feature_indices_raw.hpp>

#include <unordered_set>
#include <vector>
#include <cstdint>
#include <memory>
#include <cstring>

using std::placeholders::_1;
using FeatureIndicesRaw = lo_sam::msg::FeatureIndicesRaw;

struct RawIndex {
    int16_t scan_id;
    int16_t column_id;
};

class FeatureCloudExtractor : public rclcpp::Node
{
public:
    FeatureCloudExtractor()
    : Node("feature_cloud_extractor")
    {
        // 参数声明与获取
        this->declare_parameter<int>("n_scans", 64);
        this->declare_parameter<int>("horizon_scan", 1800);
        this->get_parameter("n_scans", N_SCANS_);
        this->get_parameter("horizon_scan", HORIZON_SCAN_);

        // 订阅
        sub_cloud_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/feature/fpga_output_cloud", 10,
            std::bind(&FeatureCloudExtractor::cloud_callback, this, _1));

        sub_flat_ = this->create_subscription<FeatureIndicesRaw>(
            "/feature/fpga_output_flat", 10,
            std::bind(&FeatureCloudExtractor::flat_callback, this, _1));
        
        sub_sharp_ = this->create_subscription<FeatureIndicesRaw>(
            "/feature/fpga_output_sharp", 10,
            std::bind(&FeatureCloudExtractor::sharp_callback, this, _1));

        sub_lesssharp_ = this->create_subscription<FeatureIndicesRaw>(
            "/feature/fpga_output_lesssharp", 10,
            std::bind(&FeatureCloudExtractor::lesssharp_callback, this, _1));

        // 发布
        pub_cloud_full_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("feature/cloud_full", 100);
        pub_cloud_flat_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("feature/cloud_flat", 100);
        pub_cloud_sharp_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("feature/cloud_sharp", 100);
        pub_cloud_lesssharp_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("feature/cloud_less_sharp", 100);
        pub_cloud_lessflat_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("feature/cloud_less_flat", 100);

        RCLCPP_INFO(this->get_logger(),
            "FeatureCloudExtractor node initialized. N_SCANS=%d, HORIZON_SCAN=%d", N_SCANS_, HORIZON_SCAN_);
    }

private:
    int N_SCANS_;
    int HORIZON_SCAN_;

    // 订阅缓存
    sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg_;
    FeatureIndicesRaw::SharedPtr flat_raw_msg_;
    FeatureIndicesRaw::SharedPtr sharp_raw_msg_;
    FeatureIndicesRaw::SharedPtr lesssharp_raw_msg_;

    // 订阅器
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_cloud_;
    rclcpp::Subscription<FeatureIndicesRaw>::SharedPtr sub_flat_, sub_sharp_, sub_lesssharp_;

    // 发布器
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_full_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_flat_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_sharp_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_lesssharp_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_lessflat_;

    // 校验索引合法性
    bool isValidIndex(int16_t scan_id, int16_t column_id) const
    {
        return (scan_id >= 0 && scan_id < N_SCANS_) &&
               (column_id >= 0 && column_id < HORIZON_SCAN_) &&
               ((scan_id + column_id) > 0);
    }

    void cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        cloud_msg_ = msg;
        try_publish();
    }

    void flat_callback(const FeatureIndicesRaw::SharedPtr msg)
    {
        flat_raw_msg_ = msg;
        try_publish();
    }

    void sharp_callback(const FeatureIndicesRaw::SharedPtr msg)
    {
        sharp_raw_msg_ = msg;
        try_publish();
    }

    void lesssharp_callback(const FeatureIndicesRaw::SharedPtr msg)
    {
        lesssharp_raw_msg_ = msg;
        try_publish();
    }

    // 解析二进制数据为索引列表
    std::vector<RawIndex> parse_raw_indices(const FeatureIndicesRaw &msg) const
    {
        std::vector<RawIndex> result;
        size_t count = msg.count;
        if (msg.data.size() < count * 4) {
            RCLCPP_WARN(this->get_logger(), "FeatureIndicesRaw data size too small for count");
            return result;
        }

        result.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            int16_t scan_id = 0;
            int16_t column_id = 0;
            // 低位字节序假设和系统一致，若有不同需改写
            std::memcpy(&scan_id, &msg.data[i * 4], 2);
            std::memcpy(&column_id, &msg.data[i * 4 + 2], 2);
            result.push_back({scan_id, column_id});
        }
        return result;
    }

    // 主要同步触发函数
    void try_publish()
    {
        if (!cloud_msg_ || !flat_raw_msg_ || !sharp_raw_msg_ || !lesssharp_raw_msg_) {
            return;
        }

        pub_cloud_full_->publish(*cloud_msg_);

        auto flat_indices = parse_raw_indices(*flat_raw_msg_);
        auto sharp_indices = parse_raw_indices(*sharp_raw_msg_);
        auto lesssharp_indices = parse_raw_indices(*lesssharp_raw_msg_);

        auto flat_cloud = extract_feature_cloud(*cloud_msg_, flat_indices);
        auto sharp_cloud = extract_feature_cloud(*cloud_msg_, sharp_indices);
        auto lesssharp_cloud = extract_feature_cloud(*cloud_msg_, lesssharp_indices);

        pub_cloud_flat_->publish(flat_cloud);
        pub_cloud_sharp_->publish(sharp_cloud);
        pub_cloud_lesssharp_->publish(lesssharp_cloud);

        auto lessflat_cloud = generate_lessflat_cloud(*cloud_msg_, flat_indices, sharp_indices, lesssharp_indices);
        pub_cloud_lessflat_->publish(lessflat_cloud);

        // 清理缓存
        cloud_msg_.reset();
        flat_raw_msg_.reset();
        sharp_raw_msg_.reset();
        lesssharp_raw_msg_.reset();
    }

    // 根据特征索引提取点云
    sensor_msgs::msg::PointCloud2 extract_feature_cloud(
        const sensor_msgs::msg::PointCloud2 &cloud_msg,
        const std::vector<RawIndex> &indices)
    {
        std::unordered_set<size_t> idx_set;
        for (const auto &feature : indices) {
            int16_t scan_id = feature.scan_id;
            int16_t column_id = feature.column_id;
            if (!isValidIndex(scan_id, column_id)) continue;
            size_t idx = static_cast<size_t>(column_id) * N_SCANS_ + scan_id;
            idx_set.insert(idx);
        }

        sensor_msgs::msg::PointCloud2 out_cloud;
        out_cloud.header = cloud_msg.header;
        out_cloud.height = 1;
        out_cloud.is_dense = cloud_msg.is_dense;
        out_cloud.is_bigendian = cloud_msg.is_bigendian;
        out_cloud.fields = cloud_msg.fields;
        out_cloud.point_step = cloud_msg.point_step;

        std::vector<uint8_t> data_out;
        data_out.reserve(idx_set.size() * cloud_msg.point_step);

        for (size_t idx : idx_set) {
            if (idx >= static_cast<size_t>(cloud_msg.width)) continue; // 安全检查
            const uint8_t *ptr = &cloud_msg.data[idx * cloud_msg.point_step];
            data_out.insert(data_out.end(), ptr, ptr + cloud_msg.point_step);
        }

        out_cloud.data = std::move(data_out);
        out_cloud.width = static_cast<uint32_t>(out_cloud.data.size() / out_cloud.point_step);
        out_cloud.row_step = out_cloud.width * out_cloud.point_step;

        return out_cloud;
    }

    // 生成 less_flat 点云，参考 FPGA 逻辑
    sensor_msgs::msg::PointCloud2 generate_lessflat_cloud(
        const sensor_msgs::msg::PointCloud2 &cloud_msg,
        const std::vector<RawIndex> &flat_indices,
        const std::vector<RawIndex> &sharp_indices,
        const std::vector<RawIndex> &lesssharp_indices)
    {
        std::unordered_set<size_t> corner_set;
        for (const auto &idx : sharp_indices) {
            if (isValidIndex(idx.scan_id, idx.column_id)) {
                corner_set.insert(static_cast<size_t>(idx.column_id) * N_SCANS_ + idx.scan_id);
            }
        }
        for (const auto &idx : lesssharp_indices) {
            if (isValidIndex(idx.scan_id, idx.column_id)) {
                corner_set.insert(static_cast<size_t>(idx.column_id) * N_SCANS_ + idx.scan_id);
            }
        }

        std::unordered_set<size_t> flat_set;
        for (const auto &idx : flat_indices) {
            if (isValidIndex(idx.scan_id, idx.column_id)) {
                flat_set.insert(static_cast<size_t>(idx.column_id) * N_SCANS_ + idx.scan_id);
            }
        }

        std::vector<uint8_t> data_out;
        data_out.reserve(cloud_msg.data.size());

        const uint32_t width = cloud_msg.width;
        for (size_t idx = 0; idx < static_cast<size_t>(width); ++idx) {
            int scan_id = static_cast<int>(idx % N_SCANS_);
            int column_id = static_cast<int>(idx / N_SCANS_);

            bool in_middle = (column_id >= 5) && (column_id < (HORIZON_SCAN_ - 6));
            if (!in_middle) continue;

            if (corner_set.count(idx) > 0) continue;
            if (flat_set.count(idx) > 0) continue;

            const uint8_t *ptr = &cloud_msg.data[idx * cloud_msg.point_step];
            data_out.insert(data_out.end(), ptr, ptr + cloud_msg.point_step);
        }

        sensor_msgs::msg::PointCloud2 out_cloud;
        out_cloud.header = cloud_msg.header;
        out_cloud.height = 1;
        out_cloud.width = static_cast<uint32_t>(data_out.size() / cloud_msg.point_step);
        out_cloud.row_step = out_cloud.width * cloud_msg.point_step;
        out_cloud.is_dense = cloud_msg.is_dense;
        out_cloud.is_bigendian = cloud_msg.is_bigendian;
        out_cloud.fields = cloud_msg.fields;
        out_cloud.point_step = cloud_msg.point_step;
        out_cloud.data = std::move(data_out);

        return out_cloud;
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<FeatureCloudExtractor>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
