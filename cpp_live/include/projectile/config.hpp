#pragma once

#include <string>

namespace projectile {

struct Config {
	int camera_index = 0; int width = 0; int height = 0; double fps = 0.0;
	std::string video_path;
	bool display = true; bool headless = false; bool enable_roi = false; double roi_percent = 1.0; double roi_anchor_x = .5; double roi_anchor_y = .5;
	bool clahe = true; double clahe_clip = 3.0; int clahe_tile = 8; bool tophat = true; int tophat_kernel = 11; int tophat_threshold = 12; std::string tophat_mode = "both";
	bool frame_diff = true; int frame_diff_gap = 2; int frame_diff_threshold = 25; bool bg_sub = true; int mog_history = 500; double mog_var_threshold = 40.0; bool mog_shadows = false;
	std::string combine_mode = "tophat_and_any"; bool morph_clean = true; int morph_kernel = 2;
	bool blob_filter = true; double min_area = 2.0; double max_area = 80.0; double min_circularity = .6; double min_solidity = .6; double max_aspect_ratio = 3.0;
	bool isolation_filter = true; double isolation_radius = 150.0; int isolation_min_cluster = 5;
	double kalman_max_distance = 80.0; int kalman_max_missing = 6; double kalman_max_innovation = 30.0; bool kalman_gravity = false; double gravity = .5;
	int trail_length = 60; double arc_min_span_ratio = .2; double arc_max_residual = 15.0; int arc_min_points = 8; int extrapolation_stop = 400;
	double process_scale = .75; int frame_skip = 1; int max_width = 1280; int max_height = 720; bool verbose_metrics = true;
};

Config load_config(const std::string& path, const Config& defaults = {});
void validate_config(const Config& config);

}
