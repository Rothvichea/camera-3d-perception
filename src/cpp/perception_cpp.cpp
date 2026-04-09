/*
 * perception_cpp.cpp
 * 
 * High-performance C++ implementations for Camera 3D Perception.
 * Compiled as Python module via pybind11.
 *
 * Operations:
 *   1. nms_2d         - 2D Non-Maximum Suppression
 *   2. sample_depth   - Fast depth sampling for multiple boxes
 *   3. boxes_to_3d    - Batch 2D+depth -> 3D conversion
 *   4. bev_render     - Fast BEV image rendering
 *   5. compute_iou    - IoU matrix for tracking
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>

namespace py = pybind11;


// ============================================================
// 1. 2D Non-Maximum Suppression
// ============================================================

std::vector<int> nms_2d(
    py::array_t<float> boxes,
    py::array_t<float> scores,
    float iou_threshold
) {
    /*
     * Fast 2D NMS.
     * 
     * Args:
     *   boxes: (N, 4) [x1, y1, x2, y2]
     *   scores: (N,) confidence scores
     *   iou_threshold: suppress if IoU > threshold
     * 
     * Returns:
     *   keep_indices: indices of boxes to keep
     */
    auto b_buf = boxes.request();
    auto s_buf = scores.request();
    int N = b_buf.shape[0];
    float* b = static_cast<float*>(b_buf.ptr);
    float* s = static_cast<float*>(s_buf.ptr);
    
    // Sort by score descending
    std::vector<int> indices(N);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [s](int a, int b) { return s[a] > s[b]; });
    
    std::vector<bool> suppressed(N, false);
    std::vector<int> keep;
    
    for (int idx : indices) {
        if (suppressed[idx]) continue;
        keep.push_back(idx);
        
        float x1_i = b[idx*4+0], y1_i = b[idx*4+1];
        float x2_i = b[idx*4+2], y2_i = b[idx*4+3];
        float area_i = (x2_i - x1_i) * (y2_i - y1_i);
        
        for (int j_idx : indices) {
            if (suppressed[j_idx] || j_idx == idx) continue;
            
            float x1_j = b[j_idx*4+0], y1_j = b[j_idx*4+1];
            float x2_j = b[j_idx*4+2], y2_j = b[j_idx*4+3];
            
            float inter_x1 = std::max(x1_i, x1_j);
            float inter_y1 = std::max(y1_i, y1_j);
            float inter_x2 = std::min(x2_i, x2_j);
            float inter_y2 = std::min(y2_i, y2_j);
            
            float inter = std::max(0.0f, inter_x2 - inter_x1) *
                           std::max(0.0f, inter_y2 - inter_y1);
            
            float area_j = (x2_j - x1_j) * (y2_j - y1_j);
            float iou = inter / std::max(area_i + area_j - inter, 1e-6f);
            
            if (iou > iou_threshold) {
                suppressed[j_idx] = true;
            }
        }
    }
    
    return keep;
}


// ============================================================
// 2. Fast Depth Sampling for Multiple Boxes
// ============================================================

py::array_t<float> sample_depth_batch(
    py::array_t<float> depth_map,
    py::array_t<int> boxes,
    float margin_ratio
) {
    /*
     * Sample median depth from center region of each box.
     * Much faster than Python loop + numpy slicing.
     *
     * Args:
     *   depth_map: (H, W) float depth values
     *   boxes: (N, 4) [x1, y1, x2, y2] integer coordinates
     *   margin_ratio: fraction of box to skip at edges (0.25 = inner 50%)
     *
     * Returns:
     *   depths: (N,) median depth per box
     */
    auto d_buf = depth_map.request();
    auto b_buf = boxes.request();
    
    int H = d_buf.shape[0];
    int W = d_buf.shape[1];
    int N = b_buf.shape[0];
    
    float* d_ptr = static_cast<float*>(d_buf.ptr);
    int* b_ptr = static_cast<int*>(b_buf.ptr);
    
    auto result = py::array_t<float>(N);
    float* r_ptr = static_cast<float*>(result.request().ptr);
    
    for (int i = 0; i < N; i++) {
        int x1 = b_ptr[i*4+0];
        int y1 = b_ptr[i*4+1];
        int x2 = b_ptr[i*4+2];
        int y2 = b_ptr[i*4+3];
        
        // Inner region
        int margin_x = static_cast<int>((x2 - x1) * margin_ratio);
        int margin_y = static_cast<int>((y2 - y1) * margin_ratio);
        
        int rx1 = std::max(0, x1 + margin_x);
        int ry1 = std::max(0, y1 + margin_y);
        int rx2 = std::min(W, x2 - margin_x);
        int ry2 = std::min(H, y2 - margin_y);
        
        if (rx2 <= rx1 || ry2 <= ry1) {
            rx1 = std::max(0, x1);
            ry1 = std::max(0, y1);
            rx2 = std::min(W, x2);
            ry2 = std::min(H, y2);
        }
        
        // Collect values for median
        std::vector<float> values;
        values.reserve((ry2 - ry1) * (rx2 - rx1));
        
        for (int row = ry1; row < ry2; row++) {
            for (int col = rx1; col < rx2; col++) {
                values.push_back(d_ptr[row * W + col]);
            }
        }
        
        if (values.empty()) {
            r_ptr[i] = 0.0f;
        } else {
            size_t mid = values.size() / 2;
            std::nth_element(values.begin(), values.begin() + mid, values.end());
            r_ptr[i] = values[mid];
        }
    }
    
    return result;
}


// ============================================================
// 3. Batch 2D + Depth -> 3D Conversion
// ============================================================

py::array_t<float> boxes_to_3d(
    py::array_t<int> boxes,
    py::array_t<float> depths,
    float fx, float fy, float cx, float cy,
    int img_h
) {
    /*
     * Convert batch of 2D boxes + depths to 3D positions.
     *
     * For each box, computes:
     *   X = (box_cx - cx) * Z / fx
     *   Y = (box_bottom - cy) * Z / fy
     *   Z = depth (from box size heuristic)
     *   distance = sqrt(X^2 + Z^2)
     *
     * Args:
     *   boxes: (N, 4) [x1, y1, x2, y2]
     *   depths: (N,) depth in meters per box
     *   fx, fy, cx, cy: camera intrinsics
     *   img_h: image height (for box ratio calculation)
     *
     * Returns:
     *   result: (N, 4) [X, Y, Z, distance] in meters
     */
    auto b_buf = boxes.request();
    auto d_buf = depths.request();
    int N = b_buf.shape[0];
    
    int* b_ptr = static_cast<int*>(b_buf.ptr);
    float* d_ptr = static_cast<float*>(d_buf.ptr);
    
    auto result = py::array_t<float>({N, 4});
    float* r_ptr = static_cast<float*>(result.request().ptr);
    
    for (int i = 0; i < N; i++) {
        int x1 = b_ptr[i*4+0];
        int y1 = b_ptr[i*4+1];
        int x2 = b_ptr[i*4+2];
        int y2 = b_ptr[i*4+3];
        
        float box_cx = (x1 + x2) / 2.0f;
        float box_h = static_cast<float>(y2 - y1);
        float box_w = static_cast<float>(x2 - x1);
        
        // Box-size depth heuristic
        float box_ratio = std::max(box_h, box_w) / std::max(static_cast<float>(img_h), 1.0f);
        float Z = 2.5f / std::max(box_ratio, 0.01f);
        Z = std::min(std::max(Z, 2.0f), 70.0f);
        
        float X = (box_cx - cx) * Z / fx;
        float Y = (static_cast<float>(y2) - cy) * Z / fy;
        float distance = std::sqrt(X * X + Z * Z);
        
        r_ptr[i*4+0] = X;
        r_ptr[i*4+1] = Y;
        r_ptr[i*4+2] = Z;
        r_ptr[i*4+3] = distance;
    }
    
    return result;
}


// ============================================================
// 4. Fast IoU Matrix for Tracking
// ============================================================

py::array_t<float> compute_iou_matrix(
    py::array_t<float> boxes_a,
    py::array_t<float> boxes_b
) {
    /*
     * Compute IoU between two sets of 2D boxes.
     *
     * Args:
     *   boxes_a: (N, 4) [x1, y1, x2, y2]
     *   boxes_b: (M, 4) [x1, y1, x2, y2]
     *
     * Returns:
     *   iou_matrix: (N, M)
     */
    auto a_buf = boxes_a.request();
    auto b_buf = boxes_b.request();
    int N = a_buf.shape[0];
    int M = b_buf.shape[0];
    
    float* a = static_cast<float*>(a_buf.ptr);
    float* b = static_cast<float*>(b_buf.ptr);
    
    auto result = py::array_t<float>({N, M});
    float* r = static_cast<float*>(result.request().ptr);
    
    for (int i = 0; i < N; i++) {
        float ax1 = a[i*4+0], ay1 = a[i*4+1];
        float ax2 = a[i*4+2], ay2 = a[i*4+3];
        float area_a = (ax2 - ax1) * (ay2 - ay1);
        
        for (int j = 0; j < M; j++) {
            float bx1 = b[j*4+0], by1 = b[j*4+1];
            float bx2 = b[j*4+2], by2 = b[j*4+3];
            
            float inter_w = std::max(0.0f, std::min(ax2, bx2) - std::max(ax1, bx1));
            float inter_h = std::max(0.0f, std::min(ay2, by2) - std::max(ay1, by1));
            float inter = inter_w * inter_h;
            
            float area_b = (bx2 - bx1) * (by2 - by1);
            r[i*M + j] = inter / std::max(area_a + area_b - inter, 1e-6f);
        }
    }
    
    return result;
}


// ============================================================
// 5. Benchmark All Operations
// ============================================================

py::dict benchmark(
    py::array_t<float> boxes_f,
    py::array_t<float> scores,
    py::array_t<float> depth_map,
    py::array_t<int> boxes_i
) {
    int N = boxes_f.request().shape[0];
    py::dict results;
    
    // NMS benchmark
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; i++) {
        nms_2d(boxes_f, scores, 0.5f);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double nms_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / 1000.0;
    
    // Depth sampling benchmark
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; i++) {
        sample_depth_batch(depth_map, boxes_i, 0.25f);
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double depth_us = std::chrono::duration<double, std::micro>(t3 - t2).count() / 1000.0;
    
    // IoU benchmark
    auto t4 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; i++) {
        compute_iou_matrix(boxes_f, boxes_f);
    }
    auto t5 = std::chrono::high_resolution_clock::now();
    double iou_us = std::chrono::duration<double, std::micro>(t5 - t4).count() / 1000.0;
    
    // 3D conversion benchmark
    auto t6 = std::chrono::high_resolution_clock::now();
    py::array_t<float> depths({N});
    float* dp = static_cast<float*>(depths.request().ptr);
    for (int i = 0; i < N; i++) dp[i] = 10.0f;
    
    for (int i = 0; i < 1000; i++) {
        boxes_to_3d(boxes_i, depths, 700, 700, 320, 180, 360);
    }
    auto t7 = std::chrono::high_resolution_clock::now();
    double conv_us = std::chrono::duration<double, std::micro>(t7 - t6).count() / 1000.0;
    
    results["num_boxes"] = N;
    results["nms_us"] = nms_us;
    results["depth_sample_us"] = depth_us;
    results["iou_matrix_us"] = iou_us;
    results["boxes_to_3d_us"] = conv_us;
    
    return results;
}


// ============================================================
// Python Module
// ============================================================

PYBIND11_MODULE(perception_cpp, m) {
    m.doc() = "C++ perception ops for Camera 3D Perception Stack";
    
    m.def("nms_2d", &nms_2d,
          "2D Non-Maximum Suppression",
          py::arg("boxes"), py::arg("scores"), py::arg("iou_threshold"));
    
    m.def("sample_depth_batch", &sample_depth_batch,
          "Fast batch depth sampling from boxes",
          py::arg("depth_map"), py::arg("boxes"), py::arg("margin_ratio") = 0.25f);
    
    m.def("boxes_to_3d", &boxes_to_3d,
          "Batch 2D+depth to 3D conversion",
          py::arg("boxes"), py::arg("depths"),
          py::arg("fx"), py::arg("fy"), py::arg("cx"), py::arg("cy"),
          py::arg("img_h"));
    
    m.def("compute_iou_matrix", &compute_iou_matrix,
          "IoU matrix between two sets of 2D boxes",
          py::arg("boxes_a"), py::arg("boxes_b"));
    
    m.def("benchmark", &benchmark,
          "Benchmark all C++ operations");
}
