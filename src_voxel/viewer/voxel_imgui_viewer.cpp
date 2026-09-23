#include "include_voxel/viewer/voxel_imgui_viewer.h"
#include "include_voxel/viewer/voxel_camera_navigation.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>

namespace
{
glm::mat4 cameraViewFromTcw(const Sophus::SE3f& Tcw)
{
    glm::mat4 cv_to_opengl(1.0f);
    cv_to_opengl[1][1] = -1.0f;
    cv_to_opengl[2][2] = -1.0f;
    return cv_to_opengl * trans4x4Eigen2glm(Tcw.matrix());
}
} // namespace

static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "[VoxelImGuiViewer]GLFW Error %d: %s\n", error, description);
}

VoxelImGuiViewer::VoxelImGuiViewer(
    std::shared_ptr<ORB_SLAM3::System> pSLAM,
    std::shared_ptr<VoxelMapper> pVoxelMapper,
    bool training)
    : glfw_window_width_(1600),
      glfw_window_height_(900),
      panel_width_(372),
      display_panel_height_(232),
      SLAM_image_viewer_scale_(1.0f),
      training_(training)
{
    this->pSLAM_ = pSLAM;
    this->pVoxelMapper_ = pVoxelMapper;

    cv::Size im_size;
    if (pSLAM)
    {
        // ORB_SLAM3 settings
        ORB_SLAM3::Settings* settings = pSLAM->getSettings();

        im_size = settings->newImSize();
        image_height_ = im_size.height;
        image_width_ = im_size.width;

        viewpointX_ = settings->viewPointX();
        viewpointY_ = settings->viewPointY();
        viewpointZ_ = settings->viewPointZ();
        viewpointF_ = settings->camera1()->getParameter(1);
    }
    else
    {
        image_height_ = pVoxelMapper->scene_->cameras_.begin()->second.height_;
        image_width_ = pVoxelMapper->scene_->cameras_.begin()->second.width_;
        viewpointF_ = pVoxelMapper->scene_->cameras_.begin()->second.params_[1];
    }

    main_fx_ = pVoxelMapper->scene_->cameras_.begin()->second.params_[0];
    main_fy_ = pVoxelMapper->scene_->cameras_.begin()->second.params_[1];
    main_cx_ = pVoxelMapper->scene_->cameras_.begin()->second.params_[2];
    main_cy_ = pVoxelMapper->scene_->cameras_.begin()->second.params_[3];

    // Voxel mapper settings
    std::filesystem::path cfg_file_path = pVoxelMapper->config_file_path_;
    readConfigFromFile(cfg_file_path);
    if (const char* preset = std::getenv("VOXEL_VIEWER_PRESET"))
    {
        const std::string name(preset);
        if (name != "default" && name != "orb" && name != "voxel" &&
            name != "tracking" && name != "reconstruction" && name != "realworld")
            throw std::invalid_argument(
                "VOXEL_VIEWER_PRESET must be default, orb, voxel, tracking, or realworld");
        applyRecordingPreset(name == "orb" ? 1 : name == "voxel" ? 2 :
                             name == "tracking" ? 3 : name == "realworld" ? 4 :
                             name == "reconstruction" ? 5 : 0);
        if (recording_preset_)
        {
            glfw_window_width_ = 1600;
            glfw_window_height_ = 900;
        }
        std::cout << "[VoxelImGuiViewer] Recording preset: " << name << std::endl;
    }
    SLAM_image_viewer_scale_ = static_cast<float>(rendered_image_width_) / image_width_;

    constexpr float near_plane = 0.01f;
    constexpr float far_plane = 1000.0f;
    const float left = -main_cx_ * near_plane / main_fx_;
    const float right =
        (static_cast<float>(image_width_) - main_cx_) *
        near_plane / main_fx_;
    const float top = main_cy_ * near_plane / main_fy_;
    const float bottom =
        -(static_cast<float>(image_height_) - main_cy_) *
        near_plane / main_fy_;
    cam_proj_ = glm::frustum(
        left,
        right,
        bottom,
        top,
        near_plane,
        far_plane);

    up_ = glm::vec3(0.0f, -1.0f, 0.0f);
    up_aligned_ = glm::vec4(up_, 1.0f);
    behind_ = glm::vec4(0.0f, 0.0f, -camera_watch_dist_, 1.0f);
    cam_pos_ = glm::vec3(viewpointX_, viewpointY_, viewpointZ_);
    cam_target_ = glm::vec3(0.0f, 0.0f, 0.0f);
    cam_view_ = glm::lookAt(cam_pos_, cam_target_, up_);
    cam_trans_ = cam_proj_ * cam_view_;

    // Create drawers
    if (pSLAM)
    {
        pSlamFrameDrawer_ = pSLAM->getFrameDrawer();
        pSlamMapDrawer_ = pSLAM->getMapDrawer();
        pMapDrawer_ = std::make_shared<ORB_SLAM3::VoxelMapDrawer>(
            pSLAM->getAtlas(), std::string(), pSLAM->getSettings());
    }
}

void VoxelImGuiViewer::readConfigFromFile(std::filesystem::path cfg_path)
{
    cv::FileStorage settings_file(cfg_path.string().c_str(), cv::FileStorage::READ);
    if(!settings_file.isOpened())
       throw std::runtime_error("[VoxelImGuiViewer]Failed to open settings file at: " + cfg_path.string());
    std::cout << "[VoxelImGuiViewer]Reading parameters from " << cfg_path << std::endl;

    glfw_window_width_ =
        settings_file["VoxelViewer.glfw_window_width"].operator int();
    glfw_window_height_ =
        settings_file["VoxelViewer.glfw_window_height"].operator int();
    rendered_image_viewer_scale_ =
        settings_file["VoxelViewer.image_scale"].operator float();
    rendered_image_height_ = image_height_ * rendered_image_viewer_scale_;
    rendered_image_width_ = image_width_ * rendered_image_viewer_scale_;

    int temp = rendered_image_width_ % 4;
    padded_sub_image_width_ = rendered_image_width_ + 4 - (temp == 0 ? 4 : temp);

    rendered_image_viewer_scale_main_ =
        settings_file["VoxelViewer.image_scale_main"].operator float();
    rendered_image_height_main_ = image_height_ * rendered_image_viewer_scale_main_;
    rendered_image_width_main_ = image_width_ * rendered_image_viewer_scale_main_;

    temp = rendered_image_width_main_ % 4;
    padded_main_image_width_ = rendered_image_width_main_ + 4 - (temp == 0 ? 4 : temp); 

    camera_watch_dist_ =
        settings_file["VoxelViewer.camera_watch_dist"].operator float();

    // Initialize configurations same as the VoxelMapper
    geo_lr_ = pVoxelMapper_->geoLearningRateInit();
    sh0_lr_ = pVoxelMapper_->sh0LearningRate();
    shs_lr_ = pVoxelMapper_->shsLearningRate();
    lambda_ssim_ = pVoxelMapper_->lambdaSsim();
    densify_interval_ = pVoxelMapper_->densifyInterval();
    new_kf_times_of_use_ = pVoxelMapper_->newKeyframeTimesOfUse();
    stable_num_iter_existence_ = pVoxelMapper_->stableNumIterExistence();
    do_gaus_pyramid_training_ = pVoxelMapper_->isdoingGausPyramidTraining();
}

void VoxelImGuiViewer::applyRecordingPreset(int preset)
{
    recording_preset_ = preset;
    fit_zoom_ = preset == 4 ? 1.2f : 1.0f;
    auto_fit_reconstruction_ = preset == 1 || preset == 2 || preset == 4 || preset == 5;
    last_auto_fit_time_ = -1.0;
    show_slam_frame_ = preset == 1 || preset == 2 || preset == 4;
    show_rendered_frame_ = preset == 2 || preset == 4;
    show_sparse_mappoints_ = preset == 1;
    show_main_rendered_ = preset != 1;
    show_keyframes_ = preset != 3;
    show_trajectory_ = preset != 3;
    tracking_vision_ = preset == 3;
    show_display_controls_ = false;
    fit_reconstruction_requested_ = preset != 3;
}

void VoxelImGuiViewer::run()
{
    // Initialize glfw
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        throw std::runtime_error("[VoxelImGuiViewer]Fails to initialize!");

    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

    // Create window with graphics context
    GLFWwindow* window =
        glfwCreateWindow(glfw_window_width_, glfw_window_height_,
                         "Photo-SLAM SVRecon", nullptr, nullptr);
    if (window == nullptr)
        throw std::runtime_error("[VoxelImGuiViewer]Fails to create window!");
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync
    glEnable(GL_DEPTH_TEST); // Enable 3D Mouse handler

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.Fonts->AddFontDefault();
    ImFontConfig frame_title_config;
    frame_title_config.SizePixels = 22.0f;
    ImFont* frame_title_font = nullptr;
    for (const char* font_path : {
             "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
             "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"})
    {
        if (std::filesystem::is_regular_file(font_path))
            frame_title_font = io.Fonts->AddFontFromFileTTF(font_path, 22.0f);
        if (frame_title_font)
            break;
    }
    if (!frame_title_font)
        frame_title_font = io.Fonts->AddFontDefault(&frame_title_config);
    io.IniFilename = nullptr;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;   // Enable Gamepad Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsClassic();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Variables for tracking
    Sophus::SE3f Tcw, TcwInit;
    glm::mat4 glmTwc, Twr, glmTwcInit;
    glmTwc = glm::mat4(1.0f);
    glmTwcInit = glm::mat4(1.0f);
    glmTwc_main_ = glm::mat4(1.0f);
    glm::mat4 Ow, OwInit;
    Ow = glm::mat4(1.0f);
    OwInit = glm::mat4(1.0f);

    // Variables for showing images
    cv::Rect image_rect_sub(0, 0, rendered_image_width_, rendered_image_height_);
    cv::Rect image_rect_main(0, 0, rendered_image_width_main_, rendered_image_height_main_);

    GLuint SLAM_img_texture, rendered_img_texture, main_img_texture;

    glGenTextures(1, &SLAM_img_texture);
    glBindTexture(GL_TEXTURE_2D, SLAM_img_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glGenTextures(1, &rendered_img_texture);
    glBindTexture(GL_TEXTURE_2D, rendered_img_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glGenTextures(1, &main_img_texture);
    glBindTexture(GL_TEXTURE_2D, main_img_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // Main loop
    while(!isStopped() && !glfwWindowShouldClose(window))
    {
        //--------------Poll and handle events (inputs, window resize, etc.)--------------
        glfwPollEvents();

        //--------------Start the Dear ImGui frame--------------
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        if (ImGui::IsKeyPressed(ImGuiKey_F1, false))
            show_display_controls_ = !show_display_controls_;
        if (ImGui::IsKeyPressed(ImGuiKey_F2, false))
            applyRecordingPreset(1);
        if (ImGui::IsKeyPressed(ImGuiKey_F3, false))
            applyRecordingPreset(2);
        if (ImGui::IsKeyPressed(ImGuiKey_F4, false))
            applyRecordingPreset(3);
        if (!io.WantCaptureKeyboard && ImGui::IsKeyPressed(ImGuiKey_F, false))
            fit_reconstruction_requested_ = true;

        main_view_origin_ = ImVec2(0, 0);
        main_view_size_ = io.DisplaySize;
        main_view_crop_ = ImVec2(1.0f, 1.0f);
        const float frame_panel_width = std::min(
            static_cast<float>(padded_sub_image_width_) + 12.0f,
            std::max(32.0f, io.DisplaySize.x * 0.4f));
        if (recording_preset_ == 5)
        {
            main_view_origin_ = ImVec2(0.0f, 0.0f);
            main_view_size_ = io.DisplaySize;
        }
        else if (recording_preset_)
        {
            const float panel_edge = (show_slam_frame_ || show_rendered_frame_)
                ? frame_panel_width + 12.0f : 16.0f;
            main_view_origin_ = ImVec2(std::min(panel_edge, io.DisplaySize.x * 0.65f), 16.0f);
            main_view_size_ = ImVec2(io.DisplaySize.x - main_view_origin_.x - 16.0f,
                                     io.DisplaySize.y - 32.0f);
            // Crop the native camera image to the taller viewport, without stretching it.
            const float aspect_ratio = (main_view_size_.x / main_view_size_.y) /
                (static_cast<float>(image_width_) / image_height_);
            main_view_crop_ = ImVec2(std::min(aspect_ratio, 1.0f),
                                     std::min(1.0f / aspect_ratio, 1.0f));
        }
        const glm::mat4 view_projection = glm::scale(glm::mat4(1.0f),
            glm::vec3(1.0f / main_view_crop_.x, 1.0f / main_view_crop_.y, 1.0f)) * cam_proj_;
        const ImVec2 main_uv_min((1.0f - main_view_crop_.x) * 0.5f,
                                (1.0f - main_view_crop_.y) * 0.5f);
        const ImVec2 main_uv_max(1.0f - main_uv_min.x, 1.0f - main_uv_min.y);

        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        //--------------Get pose of current tracked frame--------------
        if (pSLAM_)
        {
            if (!pMapDrawer_->mbSetInitCamera)
            {
                Sophus::SE3f initTwc = pSlamMapDrawer_->GetCurrentCameraPose();
                pMapDrawer_->SetInitCameraTwc(initTwc);
                pMapDrawer_->SetCurrentCameraTwc(initTwc);
                pMapDrawer_->mbSetInitCamera = true;
            }
            else
            {
                pMapDrawer_->SetCurrentCameraTwc(pSlamMapDrawer_->GetCurrentCameraPose());
            }
            pMapDrawer_->GetOpenGLCameraMatrix(true, Tcw, glmTwc, Ow);
            if (!init_Twc_set_)
                pMapDrawer_->GetOpenGLCameraMatrix(false, TcwInit, glmTwcInit, OwInit);
        }
        if (tracking_vision_)
        {
            navigation_center_valid_ = false;
            orbit_view_ = false;
            cam_view_ = cameraViewFromTcw(Tcw);
            cam_trans_ = view_projection * cam_view_;
        }
        else
        {
            if (reset_main_to_init_ || !init_Twc_set_)
            {
                glmTwc_main_ = glmTwcInit;
                Tcw_main_ = TcwInit;
                Twc_main_ = Tcw_main_.inverse();
                init_Twc_set_ = true;
                reset_main_to_init_ = false;
                navigation_center_valid_ = false;
                orbit_view_ = false;
            }
            else
            {
                Tcw_main_ = trans4x4glm2Sophus(glmTwc_main_).inverse();
                handleUserInput();
                glmTwc_main_ = trans4x4Eigen2glm(Tcw_main_.inverse().matrix());
            }
            cam_view_ = cameraViewFromTcw(Tcw_main_);
            cam_trans_ = view_projection * cam_view_;
        }

        // Fit before rendering so the voxel image and ORB overlays use one pose.
        if (auto_fit_reconstruction_ && !tracking_vision_ &&
            ImGui::GetTime() - last_auto_fit_time_ >= 1.0)
        {
            fit_reconstruction_requested_ = true;
            last_auto_fit_time_ = ImGui::GetTime();
        }
        if (fit_reconstruction_requested_ && fitViewToReconstruction())
        {
            tracking_vision_ = false;
            fit_reconstruction_requested_ = false;
            cam_view_ = cameraViewFromTcw(Tcw_main_);
            cam_trans_ = view_projection * cam_view_;
        }

        if (pSLAM_)
        {
            constexpr float panel_gap = 8.0f;
            const int visible_frame_panels =
                static_cast<int>(show_slam_frame_) + static_cast<int>(show_rendered_frame_);
            const float panel_height_limit = std::max(1.0f,
                (io.DisplaySize.y - 16.0f - panel_gap * (visible_frame_panels - 1)) /
                std::max(1, visible_frame_panels));
            auto showFrame = [&](GLuint texture, const cv::Mat& frame,
                                 const char* title, float window_y)
            {
                ImGui::PushFont(frame_title_font);
                ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(6, 6));
                const float content_width = std::max(1.0f, frame_panel_width - 12.0f);
                const float title_height = ImGui::CalcTextSize(
                    title, nullptr, false, content_width).y;
                const float image_height_limit = std::max(1.0f,
                    panel_height_limit - title_height - panel_gap - 12.0f);
                const float image_scale = std::min(1.0f, std::min(
                    content_width / frame.cols, image_height_limit / frame.rows));
                const ImVec2 image_size(frame.cols * image_scale, frame.rows * image_scale);
                const float panel_height = title_height + panel_gap + image_size.y + 12.0f;
                ImGui::SetNextWindowPos(ImVec2(0, window_y), ImGuiCond_Always);
                ImGui::SetNextWindowSize(ImVec2(frame_panel_width, panel_height), ImGuiCond_Always);
                const ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar |
                    ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                    ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse;
                if (ImGui::Begin(title, nullptr, flags))
                {
                    ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + content_width);
                    ImGui::TextUnformatted(title);
                    ImGui::PopTextWrapPos();
                    ImGui::SetCursorPos(ImVec2(
                        6.0f + (content_width - image_size.x) * 0.5f,
                        6.0f + title_height + panel_gap));
                    ImGui::Image((void*)(intptr_t)texture, image_size);
                }
                ImGui::End();
                ImGui::PopStyleVar();
                ImGui::PopFont();
                return panel_height;
            };

            float slam_window_height = 0;
            if (show_slam_frame_)
            {
                cv::Mat slam_frame = pSlamFrameDrawer_->DrawFrame(1.0f);
                if (SLAM_image_viewer_scale_ != 1.0f)
                    cv::resize(slam_frame, slam_frame, cv::Size(rendered_image_width_,
                        static_cast<int>(slam_frame.rows * SLAM_image_viewer_scale_)));
                cv::Mat padded_frame(slam_frame.rows, padded_sub_image_width_, CV_8UC3,
                                     cv::Scalar(0, 0, 0));
                slam_frame.copyTo(padded_frame(cv::Rect(0, 0, slam_frame.cols, slam_frame.rows)));
                glBindTexture(GL_TEXTURE_2D, SLAM_img_texture);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, padded_frame.cols, padded_frame.rows,
                            0, GL_BGR, GL_UNSIGNED_BYTE, padded_frame.data);
                slam_window_height = showFrame(
                    SLAM_img_texture, padded_frame, "ORB-SLAM3 frame", 0) + panel_gap;
            }

            if (show_rendered_frame_ || (show_main_rendered_ && tracking_vision_))
            {
                cv::Mat rendered_img = pVoxelMapper_->renderFromPose(
                    Tcw, rendered_image_width_, rendered_image_height_, false);
                cv::Mat rendered_img_to_show(rendered_image_height_, padded_sub_image_width_,
                                            CV_32FC3, cv::Scalar(0, 0, 0));
                rendered_img.copyTo(rendered_img_to_show(image_rect_sub));
                glBindTexture(GL_TEXTURE_2D, rendered_img_texture);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, rendered_img_to_show.cols, rendered_img_to_show.rows,
                            0, GL_RGB, GL_FLOAT, rendered_img_to_show.data);
                if (show_rendered_frame_)
                    showFrame(rendered_img_texture, rendered_img_to_show,
                              "Current Rendered Frame", slam_window_height);
            }
        }

        //--------------Draw main window image--------------
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        // Draw main window image
        if (show_main_rendered_)
        {
            auto drawlist = ImGui::GetBackgroundDrawList();
            if (pSLAM_ && tracking_vision_)
            {
                drawlist->AddImage((void *)(intptr_t)rendered_img_texture, main_view_origin_,
                    ImVec2(main_view_origin_.x + main_view_size_.x,
                           main_view_origin_.y + main_view_size_.y),
                    ImVec2(main_uv_min.x * rendered_image_width_ / padded_sub_image_width_, main_uv_min.y),
                    ImVec2(main_uv_max.x * rendered_image_width_ / padded_sub_image_width_, main_uv_max.y));
            }
            else
            {
                cv::Mat main_img = pVoxelMapper_->renderFromPose(
                    Tcw_main_, rendered_image_width_main_, rendered_image_height_main_, true);
                cv::Mat main_img_to_show = cv::Mat(rendered_image_height_main_, padded_main_image_width_, CV_32FC3, cv::Vec3f(0.0f, 0.0f, 0.0f));
                main_img.copyTo(main_img_to_show(image_rect_main));
                glBindTexture(GL_TEXTURE_2D, main_img_texture);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, main_img_to_show.cols, main_img_to_show.rows,
                     0, GL_RGB, GL_FLOAT, (float*)main_img_to_show.data);
                drawlist->AddImage((void *)(intptr_t)main_img_texture, main_view_origin_,
                    ImVec2(main_view_origin_.x + main_view_size_.x,
                           main_view_origin_.y + main_view_size_.y),
                    ImVec2(main_uv_min.x * rendered_image_width_main_ / padded_main_image_width_, main_uv_min.y),
                    ImVec2(main_uv_max.x * rendered_image_width_main_ / padded_main_image_width_, main_uv_max.y));
            }
        }
        //--------------Get current parameters--------------
        VariableParameters params_in = pVoxelMapper_->getVaribleParameters();
        geo_lr_ = params_in.geo_lr;
        sh0_lr_ = params_in.sh0_lr;
        shs_lr_ = params_in.shs_lr;
        lambda_ssim_ = params_in.lambda_ssim;
        densify_interval_ = params_in.densify_interval;
        new_kf_times_of_use_ = params_in.new_kf_times_of_use;
        stable_num_iter_existence_ = params_in.stable_num_iter_existence;
        keep_training_ = params_in.keep_training;
        do_gaus_pyramid_training_ = params_in.do_gaus_pyramid_training;

        //--------------Display mode panel--------------
        const bool mapping_complete = pVoxelMapper_->isStopped();
        if (mapping_complete && !display_controls_auto_shown_ && !recording_preset_)
        {
            show_display_controls_ = true;
            display_controls_auto_shown_ = true;
        }
        if (show_display_controls_)
        {
            ImGui::SetNextWindowPos(ImVec2(glfw_window_width_ - panel_width_, 0), ImGuiCond_Once);
            ImGui::SetNextWindowSize(ImVec2(panel_width_, display_panel_height_ + 180 +
                (terminate_run_callback_ ? 32 : 0)), ImGuiCond_Once);
            ImGui::Begin("Display Mode");

            int preset = recording_preset_;
            if (ImGui::Combo("Recording layout", &preset,
                             "Default\0ORB map\0Voxel map\0Tracking vision only\0Real-world recording\0Reconstruction only\0"))
                applyRecordingPreset(preset);
            ImGui::Checkbox("Auto-fit growing map", &auto_fit_reconstruction_);

            if (training_)
            {
                ImGui::Checkbox("Tracking vision", &tracking_vision_);
                ImGui::Checkbox("Show KeyFrames", &show_keyframes_);
                ImGui::Checkbox("Show camera trajectory", &show_trajectory_);
                ImGui::Checkbox("Show sparse MapPoints", &show_sparse_mappoints_);
            }
            ImGui::Checkbox("Show SLAM Frame", &show_slam_frame_);
            ImGui::Checkbox("Show Current Rendered Frame", &show_rendered_frame_);
            ImGui::Checkbox("Show main window rendered", &show_main_rendered_);

            if (ImGui::Button("Fit Reconstruction"))
                fit_reconstruction_requested_ = true;
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Scroll: zoom. Left drag: orbit. Right/middle drag: pan.\n"
                                  "W/S: forward/back. A/D: left/right. Q/E: up/down.");
            if (ImGui::SliderFloat("Fit zoom", &fit_zoom_, 1.0f, 2.0f, "%.2fx"))
                fit_reconstruction_requested_ = true;
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("1x fits the full map. Larger values bring the scene closer\n"
                                  "and may place distant edges outside the view.");
            if (ImGui::Button("Straighten view"))
                straightenView();
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Align the vertical direction with the first keyframe.\n"
                                  "This is a camera reference, not a gravity estimate.");
            if (ImGui::Button("Center orbit on camera path"))
                centerOrbitOnKeyframes();
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Aim at the keyframe center and rotate around it.\n"
                                  "Distant voxel artifacts do not affect this center.");
            ImGui::TextUnformatted("Alt + left drag: roll\nCtrl: precision movement\nU / O: roll left / right");

            if (terminate_run_callback_)
            {
                if (ImGui::Button("Terminate Run"))
                    terminate_run_callback_();
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Exit immediately. Unsaved results will be lost.");
            }

            ImGui::Text("Viewer average FPS %.1f", io.Framerate);
            ImGui::End();
        }

        VariableParameters params_out;
        params_out.geo_lr = geo_lr_;
        params_out.sh0_lr = sh0_lr_;
        params_out.shs_lr = shs_lr_;
        params_out.lambda_ssim = lambda_ssim_;
        params_out.densify_interval = densify_interval_;
        params_out.new_kf_times_of_use = new_kf_times_of_use_;
        params_out.stable_num_iter_existence = stable_num_iter_existence_;
        params_out.keep_training = keep_training_;
        params_out.do_gaus_pyramid_training = do_gaus_pyramid_training_;
        pVoxelMapper_->setVaribleParameters(params_out);

        //--------------ImGui Rendering--------------
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        //--------------Draw main window SLAM--------------
        const float framebuffer_scale_x = display_w / io.DisplaySize.x;
        const float framebuffer_scale_y = display_h / io.DisplaySize.y;
        const int map_x = std::lround(main_view_origin_.x * framebuffer_scale_x);
        const int map_y = std::lround((io.DisplaySize.y - main_view_origin_.y - main_view_size_.y)
                                     * framebuffer_scale_y);
        const int map_w = std::lround(main_view_size_.x * framebuffer_scale_x);
        const int map_h = std::lround(main_view_size_.y * framebuffer_scale_y);
        glViewport(map_x, map_y, map_w, map_h);
        glEnable(GL_SCISSOR_TEST);
        glScissor(map_x, map_y, map_w, map_h);
        // Set relative viewpoint
        glPushMatrix();
        glMultMatrixf(&cam_trans_[0][0]);
        // Draw camera, KeyFrames and MapPoints
        if (pSLAM_ && show_keyframes_)
        {
            pMapDrawer_->DrawCurrentCamera(glmTwc);
            pMapDrawer_->DrawKeyFrames(true, false, true, false);
        }
        if (pSLAM_ && show_trajectory_)
            pMapDrawer_->DrawKeyFrameTrajectory();
        if (pSLAM_ && show_sparse_mappoints_)
        {
            pMapDrawer_->DrawMapPoints();
        }
        // Clear relative viewpoint
        glPopMatrix();
        glDisable(GL_SCISSOR_TEST);
        glViewport(0, 0, display_w, display_h);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    if (pSLAM_ && !pSLAM_->isShutDown())
        pSLAM_->Shutdown();
    else
        pVoxelMapper_->signalStop();

    if (pVoxelMapper_->isKeepingTraining())
        pVoxelMapper_->setKeepTraining(false);
}

bool VoxelImGuiViewer::isStopped()
{
    std::unique_lock<std::mutex> lock_status(this->mutex_status_);
    return this->stopped_;
}

void VoxelImGuiViewer::signalStop(const bool going_to_stop)
{
    std::unique_lock<std::mutex> lock_status(this->mutex_status_);
    this->stopped_ = going_to_stop;
}

/**
 * We modify Twc_main_ then Tcw_main_ (Sophus::SE3f) to handle mouse and keyboard inputs
 */
void VoxelImGuiViewer::handleUserInput()
{
    if (tracking_vision_)
    {
        free_view_enabled_ = false;
        return;
    }
    else
    {
        free_view_enabled_ = true;
    }

    Twc_main_ = Tcw_main_.inverse();
    if (!navigation_center_valid_)
    {
        navigation_center_ = Twc_main_.translation() +
            std::max(camera_watch_dist_, 1.0f) * Twc_main_.rotationMatrix().col(2);
        navigation_center_valid_ = true;
    }

    // Only respond to mouse inputs when not interacting with ImGui
    const ImVec2 mouse = ImGui::GetIO().MousePos;
    const bool over_map = mouse.x >= main_view_origin_.x && mouse.y >= main_view_origin_.y &&
        mouse.x < main_view_origin_.x + main_view_size_.x &&
        mouse.y < main_view_origin_.y + main_view_size_.y;
    if (over_map && !ImGui::IsAnyItemActive() && !ImGui::GetIO().WantCaptureMouse)
    {
        mouseWheel();
        mouseDrag();
    }

    // Respond to keyboard inputs
    keyboardEvent();

    if (!Twc_main_.matrix().isApprox(Tcw_main_.inverse().matrix(), 1.0e-6f))
        auto_fit_reconstruction_ = false;
    Tcw_main_ = Twc_main_.inverse();
}

void VoxelImGuiViewer::mouseWheel()
{
    const float delta = ImGui::GetIO().MouseWheel;
    if (delta != 0)
        Twc_main_.translation() = voxel_viewer_navigation::zoomPosition(
            Twc_main_.translation(), navigation_center_,
            delta * (ImGui::GetIO().KeyCtrl ? 0.2f : 1.0f));
}

void VoxelImGuiViewer::mouseDrag()
{
    float delta_rel_x = ImGui::GetIO().MouseDelta.x / main_view_size_.x;
    float delta_rel_y = ImGui::GetIO().MouseDelta.y / main_view_size_.y;
    const ImGuiIO& io = ImGui::GetIO();
    const float precision = io.KeyCtrl ? 0.2f : 1.0f;
    const bool pan = io.MouseDown[1] || io.MouseDown[2] ||
                     (io.MouseDown[0] && io.KeyShift);

    //---Rotation---
    Eigen::Vector3f eulars = Eigen::Vector3f::Zero();
    // Left held
    if (io.MouseDown[0] && !pan)
    {
        if (io.KeyAlt)
            eulars.z() = M_PI * delta_rel_x * mouse_left_sensitivity_;
        else
        {
            eulars.x() = -M_PI * delta_rel_y * mouse_left_sensitivity_;
            eulars.y() = M_PI * delta_rel_x * mouse_left_sensitivity_;
        }
        eulars *= precision;
    }

    // To rotation matrix
    Eigen::AngleAxisf roll_angle(eulars.z(), Eigen::Vector3f::UnitZ());
    Eigen::AngleAxisf yaw_angle(eulars.y(), Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf pitch_angle(eulars.x(), Eigen::Vector3f::UnitX());
    Eigen::Quaternion<float> q = roll_angle * yaw_angle * pitch_angle;
    Eigen::Matrix3f rotating = q.matrix();
    //-------------

    const Eigen::Matrix3f R = Twc_main_.rotationMatrix();
    if (pan)
    {
        const float distance = std::max(
            (Twc_main_.translation() - navigation_center_).norm(), 0.05f);
        const Eigen::Vector3f translating = voxel_viewer_navigation::panTranslation(
            R, io.MouseDelta.x * precision, io.MouseDelta.y * precision, distance,
            main_fx_ * main_view_size_.x / (image_width_ * main_view_crop_.x),
            main_fy_ * main_view_size_.y / (image_height_ * main_view_crop_.y));
        Twc_main_.translation() += translating;
        navigation_center_ += translating;
    }
    else if (orbit_view_ && io.MouseDown[0])
    {
        Twc_main_.translation() = voxel_viewer_navigation::orbitPosition(
            Twc_main_.translation(), navigation_center_, R, rotating);
    }
    Twc_main_.setRotationMatrix(R * rotating);
    if (!orbit_view_ && eulars.squaredNorm() > 0)
    {
        const float distance = (Twc_main_.translation() - navigation_center_).norm();
        navigation_center_ = Twc_main_.translation() +
            distance * Twc_main_.rotationMatrix().col(2);
    }
}

void VoxelImGuiViewer::straightenView()
{
    if (!pMapDrawer_ || !pMapDrawer_->mpAtlas)
        return;
    auto* map = pMapDrawer_->mpAtlas->GetCurrentMap();
    if (!map)
        return;
    ORB_SLAM3::KeyFrame* first = nullptr;
    for (auto* keyframe : map->GetAllKeyFrames())
        if (keyframe && !keyframe->isBad() && (!first || keyframe->mnId < first->mnId))
            first = keyframe;
    if (!first)
        return;
    Twc_main_ = Tcw_main_.inverse();
    const Eigen::Matrix3f previous = Twc_main_.rotationMatrix();
    const Eigen::Matrix3f reference = first->GetPoseInverse().rotationMatrix();
    const Eigen::Matrix3f aligned = voxel_viewer_navigation::alignedRotation(
        previous, previous.col(2), reference.col(1));
    if (navigation_center_valid_ && orbit_view_)
        Twc_main_.translation() = voxel_viewer_navigation::orbitPosition(
            Twc_main_.translation(), navigation_center_, previous, previous.transpose() * aligned);
    Twc_main_.setRotationMatrix(aligned);
    Tcw_main_ = Twc_main_.inverse();
    glmTwc_main_ = trans4x4Eigen2glm(Twc_main_.matrix());
    tracking_vision_ = false;
    auto_fit_reconstruction_ = false;
    fit_reconstruction_requested_ = false;
}

void VoxelImGuiViewer::centerOrbitOnKeyframes()
{
    if (!pMapDrawer_ || !pMapDrawer_->mpAtlas)
        return;
    auto* map = pMapDrawer_->mpAtlas->GetCurrentMap();
    if (!map)
        return;
    Eigen::Vector3f center = Eigen::Vector3f::Zero();
    size_t count = 0;
    for (auto* keyframe : map->GetAllKeyFrames())
    {
        if (!keyframe || keyframe->isBad())
            continue;
        const Eigen::Vector3f position = keyframe->GetCameraCenter();
        if (!position.allFinite())
            continue;
        center += position;
        ++count;
    }
    if (!count)
        return;
    center /= static_cast<float>(count);
    Twc_main_ = Tcw_main_.inverse();
    const Eigen::Vector3f forward = center - Twc_main_.translation();
    if (forward.norm() < 0.05f)
        return;
    const Eigen::Matrix3f previous = Twc_main_.rotationMatrix();
    Twc_main_.setRotationMatrix(voxel_viewer_navigation::alignedRotation(
        previous, forward, previous.col(1)));
    navigation_center_ = center;
    navigation_center_valid_ = true;
    orbit_view_ = true;
    Tcw_main_ = Twc_main_.inverse();
    glmTwc_main_ = trans4x4Eigen2glm(Twc_main_.matrix());
    tracking_vision_ = false;
    auto_fit_reconstruction_ = false;
    fit_reconstruction_requested_ = false;
}

bool VoxelImGuiViewer::fitViewToReconstruction()
{
    Eigen::Vector3f minimum = Eigen::Vector3f::Constant(std::numeric_limits<float>::infinity());
    Eigen::Vector3f maximum = -minimum;
    bool have_bounds = false;
    if (show_sparse_mappoints_ && !show_main_rendered_ && pMapDrawer_)
    {
        ORB_SLAM3::Map* map = pMapDrawer_->mpAtlas->GetCurrentMap();
        if (map)
        {
            for (ORB_SLAM3::MapPoint* point : map->GetAllMapPoints())
            {
                if (!point || point->isBad())
                    continue;
                const Eigen::Vector3f position = point->GetWorldPos();
                if (!position.allFinite())
                    continue;
                minimum = minimum.cwiseMin(position);
                maximum = maximum.cwiseMax(position);
                have_bounds = true;
            }
        }
    }
    else
        have_bounds = pVoxelMapper_->getCurrentVoxelBounds(minimum, maximum);
    if (!have_bounds)
        return false;

    if ((show_trajectory_ || show_keyframes_) && pMapDrawer_ && pMapDrawer_->mpAtlas)
    {
        ORB_SLAM3::Map* map = pMapDrawer_->mpAtlas->GetCurrentMap();
        if (map)
        {
            for (ORB_SLAM3::KeyFrame* keyframe : map->GetAllKeyFrames())
            {
                if (!keyframe || keyframe->isBad())
                    continue;
                const Eigen::Vector3f center = keyframe->GetCameraCenter();
                minimum = minimum.cwiseMin(center);
                maximum = maximum.cwiseMax(center);
            }
        }
    }

    if (!minimum.allFinite() || !maximum.allFinite())
        return false;
    const Eigen::Vector3f scene_center = 0.5f * (minimum + maximum);
    if (tracking_vision_ && pSlamMapDrawer_)
        Tcw_main_ = pSlamMapDrawer_->GetCurrentCameraPose();
    Twc_main_ = Tcw_main_.inverse();
    const Eigen::Matrix3f camera_to_world = Twc_main_.rotationMatrix();
    const float fit_padding = recording_preset_ == 4 || recording_preset_ == 5
        ? 1.02f : 1.08f;
    const float distance = voxel_viewer_navigation::fitDistance(
        minimum, maximum, camera_to_world,
        main_fx_ / main_view_crop_.x, main_fy_ / main_view_crop_.y,
        (main_cx_ - image_width_ * 0.5f) / main_view_crop_.x + image_width_ * 0.5f,
        (main_cy_ - image_height_ * 0.5f) / main_view_crop_.y + image_height_ * 0.5f,
        image_width_, image_height_, fit_padding);
    const Eigen::Vector3f forward = camera_to_world.col(2).normalized();
    Twc_main_.translation() = scene_center - distance / fit_zoom_ * forward;
    Tcw_main_ = Twc_main_.inverse();
    glmTwc_main_ = trans4x4Eigen2glm(Twc_main_.matrix());
    navigation_center_ = scene_center;
    navigation_center_valid_ = true;
    orbit_view_ = true;
    init_Twc_set_ = true;
    return true;
}

void VoxelImGuiViewer::keyboardEvent()
{
    if (ImGui::GetIO().WantCaptureKeyboard)
        return;

    // R: Reset camera view to init
    if (ImGui::IsKeyPressed(ImGuiKey_R))
        reset_main_to_init_ = true;

    //---Translation---
    Eigen::Vector3f translating = Eigen::Vector3f::Zero();
    // W: forward
    if (ImGui::IsKeyDown(ImGuiKey_W))
        translating.z() += 1.0f;
    // S: backward
    if (ImGui::IsKeyDown(ImGuiKey_S))
        translating.z() -= 1.0f;
    // A: leftward
    if (ImGui::IsKeyDown(ImGuiKey_A) || ImGui::IsKeyDown(ImGuiKey_LeftArrow))
        translating.x() -= 1.0f;
    // D: rightward
    if (ImGui::IsKeyDown(ImGuiKey_D) || ImGui::IsKeyDown(ImGuiKey_RightArrow))
        translating.x() += 1.0f;
    if (ImGui::IsKeyDown(ImGuiKey_Q) || ImGui::IsKeyDown(ImGuiKey_UpArrow))
        translating.y() -= 1.0f;
    if (ImGui::IsKeyDown(ImGuiKey_E) || ImGui::IsKeyDown(ImGuiKey_DownArrow))
        translating.y() += 1.0f;
    // Velocity
    const float frame_scale = std::min(ImGui::GetIO().DeltaTime, 0.1f) * 10.0f *
        (ImGui::GetIO().KeyCtrl ? 0.2f : 1.0f);
    translating *= keyboard_velocity_ * frame_scale;
    //-------------

    //---Rotation---
    Eigen::Vector3f eulars = Eigen::Vector3f::Zero();
    // I: pitch upward (Rx+)
    if (ImGui::IsKeyDown(ImGuiKey_I))
        eulars.x() += M_PI;
    // K: pitch downward (Rx-)
    if (ImGui::IsKeyDown(ImGuiKey_K))
        eulars.x() -= M_PI;
    // J: yaw leftward (Ry-)
    if (ImGui::IsKeyDown(ImGuiKey_J))
        eulars.y() -= M_PI;
    // L: yaw rightward (Ry+)
    if (ImGui::IsKeyDown(ImGuiKey_L))
        eulars.y() += M_PI;
    // U: roll counterclockwise (Rz-)
    if (ImGui::IsKeyDown(ImGuiKey_U))
        eulars.z() -= M_PI;
    // O: roll clockwise (Rz+)
    if (ImGui::IsKeyDown(ImGuiKey_O))
        eulars.z() += M_PI;
    // Angular Velocity
    eulars *= keyboard_anglular_velocity_ * frame_scale;
    // To rotation matrix
    Eigen::AngleAxisf roll_angle(eulars.z(), Eigen::Vector3f::UnitZ());
    Eigen::AngleAxisf yaw_angle(eulars.y(), Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf pitch_angle(eulars.x(), Eigen::Vector3f::UnitX());
    Eigen::Quaternion<float> q = roll_angle * yaw_angle * pitch_angle;
    Eigen::Matrix3f rotating = q.matrix();
    //--------------

    //---Apply---
    Eigen::Matrix3f R = Twc_main_.rotationMatrix();
    const Eigen::Vector3f world_translation = R * translating;
    Twc_main_.translation() += world_translation;
    navigation_center_ += world_translation;
    if (orbit_view_ && eulars.squaredNorm() > 0)
        Twc_main_.translation() = voxel_viewer_navigation::orbitPosition(
            Twc_main_.translation(), navigation_center_, R, rotating);
    Twc_main_.setRotationMatrix(R * rotating);
    if (!orbit_view_ && eulars.squaredNorm() > 0)
    {
        const float distance = (Twc_main_.translation() - navigation_center_).norm();
        navigation_center_ = Twc_main_.translation() +
            distance * Twc_main_.rotationMatrix().col(2);
    }
}
