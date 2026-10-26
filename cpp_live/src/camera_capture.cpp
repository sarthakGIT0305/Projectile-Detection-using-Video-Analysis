 #include "projectile/camera_capture.hpp"
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#include <winhttp.h>
#endif

namespace projectile {

#ifdef _WIN32
static std::string download_http_video(const std::string& url)
{
	const std::wstring wide_url(url.begin(), url.end());
	URL_COMPONENTS parts{};
	wchar_t host[256]{};
	wchar_t url_path[2048]{};
	wchar_t extra_info[2048]{};
	parts.dwStructSize = sizeof(parts);
	parts.lpszHostName = host;
	parts.dwHostNameLength = static_cast<DWORD>(std::size(host));
	parts.lpszUrlPath = url_path;
	parts.dwUrlPathLength = static_cast<DWORD>(std::size(url_path));
	parts.lpszExtraInfo = extra_info;
	parts.dwExtraInfoLength = static_cast<DWORD>(std::size(extra_info));

	if (!WinHttpCrackUrl(wide_url.c_str(), 0, 0, &parts))
		return {};

	HINTERNET session = WinHttpOpen(
		L"projectile_live/1.0", WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
		WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
	if (!session)
		return {};

	HINTERNET connection = WinHttpConnect(
		session, host, parts.nPort, 0);
	const bool secure = parts.nScheme == INTERNET_SCHEME_HTTPS;
	std::wstring target(url_path, parts.dwUrlPathLength);
	target.append(extra_info, parts.dwExtraInfoLength);
	HINTERNET request = connection ? WinHttpOpenRequest(
		connection, L"GET", target.c_str(), nullptr, WINHTTP_NO_REFERER,
		WINHTTP_DEFAULT_ACCEPT_TYPES, secure ? WINHTTP_FLAG_SECURE : 0) : nullptr;

	auto close_handles = [&]() {
		if (request) WinHttpCloseHandle(request);
		if (connection) WinHttpCloseHandle(connection);
		WinHttpCloseHandle(session);
	};

	if (!request || !WinHttpSendRequest(
			request, WINHTTP_NO_ADDITIONAL_HEADERS, 0,
			WINHTTP_NO_REQUEST_DATA, 0, 0, 0) ||
		!WinHttpReceiveResponse(request, nullptr))
	{
		close_handles();
		return {};
	}

	DWORD status = 0;
	DWORD status_size = sizeof(status);
	if (!WinHttpQueryHeaders(
			request, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
			WINHTTP_HEADER_NAME_BY_INDEX, &status, &status_size,
			WINHTTP_NO_HEADER_INDEX) || status < 200 || status >= 300)
	{
		close_handles();
		return {};
	}

	wchar_t temporary_file[MAX_PATH]{};
	wchar_t temporary_directory[MAX_PATH]{};
	if (!GetTempPathW(static_cast<DWORD>(std::size(temporary_directory)), temporary_directory) ||
		!GetTempFileNameW(temporary_directory, L"prj", 0, temporary_file))
	{
		close_handles();
		return {};
	}

	const std::filesystem::path output_path(temporary_file);
	HANDLE output = CreateFileW(
		output_path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
		FILE_ATTRIBUTE_TEMPORARY, nullptr);
	if (output == INVALID_HANDLE_VALUE)
	{
		DeleteFileW(output_path.c_str());
		close_handles();
		return {};
	}

	bool success = true;
	do
	{
		DWORD available = 0;
		if (!WinHttpQueryDataAvailable(request, &available))
		{
			success = false;
			break;
		}
		if (available == 0)
			break;

		std::vector<char> buffer(available);
		DWORD received = 0;
		if (!WinHttpReadData(request, buffer.data(), available, &received))
		{
			success = false;
			break;
		}

		DWORD written = 0;
		if (!WriteFile(output, buffer.data(), received, &written, nullptr) || written != received)
		{
			success = false;
			break;
		}
	} while (true);

	CloseHandle(output);
	close_handles();
	if (!success)
	{
		DeleteFileW(output_path.c_str());
		return {};
	}
	return output_path.string();
}
#endif

CameraCapture::CameraCapture(int i,int w,int h,double f,std::string path,LatestFrameBuffer& b):index_(i),width_(w),height_(h),fps_(f),video_path_(std::move(path)),buffer_(b){}
CameraCapture::~CameraCapture(){stop();}
void CameraCapture::start(){if(worker_.joinable())return;std::string source=video_path_;
#ifdef _WIN32
if(!video_path_.empty()&&(video_path_.rfind("http://",0)==0||video_path_.rfind("https://",0)==0)){temporary_video_path_=download_http_video(video_path_);if(temporary_video_path_.empty()){std::cerr<<"Unable to download video "<<video_path_<<"\n";return;}source=temporary_video_path_;}
#endif
bool ok=video_path_.empty()?capture_.open(index_):capture_.open(source);if(!ok){std::cerr<<(video_path_.empty()?"Unable to open camera "+std::to_string(index_):"Unable to open video "+video_path_)<<"\n";if(!temporary_video_path_.empty()){std::remove(temporary_video_path_.c_str());temporary_video_path_.clear();}return;}if(video_path_.empty()){if(width_>0)capture_.set(cv::CAP_PROP_FRAME_WIDTH,width_);if(height_>0)capture_.set(cv::CAP_PROP_FRAME_HEIGHT,height_);if(fps_>0)capture_.set(cv::CAP_PROP_FPS,fps_);}opened_=true;worker_=std::thread(&CameraCapture::run,this);}
void CameraCapture::stop(){stopping_=true;buffer_.close();if(worker_.joinable())worker_.join();if(capture_.isOpened())capture_.release();if(!temporary_video_path_.empty()){std::remove(temporary_video_path_.c_str());temporary_video_path_.clear();}opened_=false;}
void CameraCapture::run(){std::uint64_t s=0;while(!stopping_){cv::Mat f;if(!capture_.read(f)||f.empty()){std::cerr<<(video_path_.empty()?"Camera read failed or disconnected":"Video playback finished or failed")<<"\n";break;}FramePacket packet{f,s++,std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count()};if(video_path_.empty()){buffer_.push(std::move(packet));}else if(!buffer_.wait_push(std::move(packet),stopping_)){break;}++captured_;}buffer_.close();}
}
