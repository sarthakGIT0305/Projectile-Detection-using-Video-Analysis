#include "projectile/config.hpp"
#include <fstream>
#include <stdexcept>
#include <string>
namespace projectile {
static std::string trim(std::string s){auto a=s.find_first_not_of(" \t\r\n"),b=s.find_last_not_of(" \t\r\n");return a==std::string::npos?"":s.substr(a,b-a+1);}
static bool boolean(const std::string& v){return v=="1"||v=="true"||v=="on"||v=="yes";}
Config load_config(const std::string& path,const Config& defaults){Config c=defaults;std::ifstream in(path);if(!in)throw std::runtime_error("cannot open config: "+path);std::string line;while(std::getline(in,line)){line=trim(line);if(line.empty()||line[0]=='#'||line[0]==';'||line[0]=='[')continue;auto p=line.find('=');if(p==std::string::npos)continue;auto k=trim(line.substr(0,p)),v=trim(line.substr(p+1));try{
#define I(name) else if(k==#name)c.name=std::stoi(v)
#define D(name) else if(k==#name)c.name=std::stod(v)
#define B(name) else if(k==#name)c.name=boolean(v)
 if(k=="camera_index")c.camera_index=std::stoi(v);else if(k=="video_path")c.video_path=v;I(width);I(height);D(fps);B(display);B(headless);D(process_scale);D(display_scale);I(frame_skip);B(enable_roi);D(roi_percent);D(roi_anchor_x);D(roi_anchor_y);B(clahe);D(clahe_clip);I(clahe_tile);B(tophat);I(tophat_kernel);I(tophat_threshold);else if(k=="tophat_mode")c.tophat_mode=v;B(frame_diff);I(frame_diff_gap);I(frame_diff_threshold);B(bg_sub);I(mog_history);D(mog_var_threshold);B(mog_shadows);else if(k=="combine_mode")c.combine_mode=v;B(morph_clean);I(morph_kernel);B(blob_filter);D(min_area);D(max_area);D(min_circularity);D(min_solidity);D(max_aspect_ratio);B(isolation_filter);D(isolation_radius);I(isolation_min_cluster);D(kalman_max_distance);I(kalman_max_missing);D(kalman_max_innovation);B(kalman_gravity);D(gravity);I(trail_length);D(arc_min_span_ratio);D(arc_max_residual);I(arc_min_points);I(extrapolation_stop);B(verbose_metrics);
#undef I
#undef D
#undef B
}catch(const std::exception&){throw std::runtime_error("invalid value for config key: "+k);}}validate_config(c);return c;}
void validate_config(const Config& c){if(c.frame_diff_gap<1||c.morph_kernel<1||c.tophat_kernel<1||c.min_area<=0||c.min_area>=c.max_area||c.process_scale<=0||c.process_scale>1||c.display_scale<=0||c.display_scale>1||c.kalman_max_missing<0)throw std::invalid_argument("invalid projectile configuration");}
}
