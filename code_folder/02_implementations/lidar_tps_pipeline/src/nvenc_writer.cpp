// nvenc_writer.cpp -- see nvenc_writer.hpp for the architecture overview.

#include "nvenc_writer.hpp"
#include "bgr_to_nv12.cuh"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
}

#include <stdexcept>
#include <string>

namespace lidartps {

namespace {

[[noreturn]] static void die(const std::string& what, int err) {
    char buf[256];
    av_strerror(err, buf, sizeof(buf));
    throw std::runtime_error("nvenc_writer: " + what + ": " + buf);
}

#define AV_OK(call, what) do { int _r = (call); if (_r < 0) die(what, _r); } while(0)

} // namespace

struct NvencWriter::Impl {
    AVFormatContext* fmt        = nullptr;
    AVStream*        stream     = nullptr;
    AVCodecContext*  enc        = nullptr;

    AVBufferRef*     hw_dev_ref = nullptr;   // CUDA hwdevice
    AVBufferRef*     hw_frm_ref = nullptr;   // CUDA hwframes (NV12 pool)

    AVPacket*        pkt        = nullptr;

    int   W = 0, H = 0;       // even-rounded dims
    int   src_W = 0, src_H = 0;
    int64_t pts = 0;
    bool  open = false;
    bool  y_only = false;     // chroma-flattened encode for the bitrate-weight calibration
};

NvencWriter::NvencWriter() : d_(new Impl) {}
NvencWriter::~NvencWriter() {
    close();
    delete d_;
}

bool NvencWriter::isOpen() const { return d_ && d_->open; }

void NvencWriter::open(const std::string& path, int width, int height, int fps,
                       bool y_only) {
    auto& d = *d_;
    if (d.open) close();
    d.y_only = y_only;

    d.src_W = width;
    d.src_H = height;
    // NVENC requires even dimensions for chroma subsampling. Round up; the
    // bottom row gets duplicated by the BGR->NV12 kernel via clamp-to-edge,
    // but in practice we just pad by writing padding-row zeros via the host
    // allocation -- matches what cv::VideoWriter did previously.
    d.W = (width  + 1) & ~1;
    d.H = (height + 1) & ~1;

    // -- output container ----------------------------------------------------
    AV_OK(avformat_alloc_output_context2(&d.fmt, nullptr, nullptr, path.c_str()),
          "avformat_alloc_output_context2");

    // HEVC NVENC -- caps at 8192x8192 (vs H.264's 4096), fits our 5238x2304.
    const AVCodec* codec = avcodec_find_encoder_by_name("hevc_nvenc");
    if (!codec) throw std::runtime_error("hevc_nvenc not available");

    d.stream = avformat_new_stream(d.fmt, nullptr);
    d.enc    = avcodec_alloc_context3(codec);
    d.enc->codec_id    = codec->id;
    d.enc->width       = d.W;
    d.enc->height      = d.H;
    d.enc->pix_fmt     = AV_PIX_FMT_CUDA;          // GPU-resident frames
    d.enc->sw_pix_fmt  = AV_PIX_FMT_NV12;          // backed by NV12
    d.enc->time_base   = AVRational{1, fps};
    d.enc->framerate   = AVRational{fps, 1};
    d.enc->gop_size    = fps;                       // 1 keyframe / s
    d.enc->max_b_frames = 0;
    av_opt_set(d.enc->priv_data, "preset", "p4", 0);  // medium quality / perf
    av_opt_set(d.enc->priv_data, "tune",   "ll", 0);  // low-latency
    av_opt_set(d.enc->priv_data, "rc",     "vbr", 0);
    av_opt_set_int(d.enc->priv_data, "cq", 23, 0);
    if (d.fmt->oformat->flags & AVFMT_GLOBALHEADER)
        d.enc->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    // -- CUDA hwdevice + hwframes pool (NV12 on GPU) ------------------------
    AV_OK(av_hwdevice_ctx_create(&d.hw_dev_ref, AV_HWDEVICE_TYPE_CUDA,
                                 nullptr, nullptr, 0),
          "av_hwdevice_ctx_create(CUDA)");

    d.hw_frm_ref = av_hwframe_ctx_alloc(d.hw_dev_ref);
    if (!d.hw_frm_ref) throw std::runtime_error("av_hwframe_ctx_alloc failed");
    auto* frames_ctx = (AVHWFramesContext*)d.hw_frm_ref->data;
    frames_ctx->format    = AV_PIX_FMT_CUDA;
    frames_ctx->sw_format = AV_PIX_FMT_NV12;
    frames_ctx->width     = d.W;
    frames_ctx->height    = d.H;
    frames_ctx->initial_pool_size = 4;   // small ring, single-producer
    AV_OK(av_hwframe_ctx_init(d.hw_frm_ref), "av_hwframe_ctx_init");

    d.enc->hw_frames_ctx = av_buffer_ref(d.hw_frm_ref);

    AV_OK(avcodec_open2(d.enc, codec, nullptr), "avcodec_open2");
    AV_OK(avcodec_parameters_from_context(d.stream->codecpar, d.enc),
          "avcodec_parameters_from_context");
    d.stream->time_base = d.enc->time_base;

    if (!(d.fmt->oformat->flags & AVFMT_NOFILE))
        AV_OK(avio_open(&d.fmt->pb, path.c_str(), AVIO_FLAG_WRITE), "avio_open");
    AV_OK(avformat_write_header(d.fmt, nullptr), "avformat_write_header");

    d.pkt = av_packet_alloc();
    d.pts = 0;
    d.open = true;
}

void NvencWriter::writeFrame(const uint8_t* d_bgr, cudaStream_t stream) {
    auto& d = *d_;
    if (!d.open) throw std::runtime_error("NvencWriter::writeFrame on closed writer");

    // Get a fresh GPU frame from the hwframes pool. The previous frame's
    // buffer is still owned by the encoder (the AVCodecContext keeps a
    // reference until the encoder finishes reading); reusing the same
    // buffer would race against NVENC. With pool_size=4, up to 4 frames
    // can be in flight before this call back-pressures.
    AVFrame* fr = av_frame_alloc();
    AV_OK(av_hwframe_get_buffer(d.hw_frm_ref, fr, 0), "av_hwframe_get_buffer");
    fr->width  = d.W;
    fr->height = d.H;

    auto* y_plane  = (uint8_t*)fr->data[0];
    auto* uv_plane = (uint8_t*)fr->data[1];
    int   y_stride  = fr->linesize[0];
    int   uv_stride = fr->linesize[1];

    bgrToNv12(d_bgr, y_plane, uv_plane,
              d.src_W, d.src_H,    // logical pipeline dims (may be odd)
              y_stride, uv_stride,
              stream,
              d.y_only);

    // Make sure the kernel has finished before NVENC reads. The encoder lives
    // on its own libav-managed CUDA stream; without the sync, NVENC could
    // start reading mid-conversion.
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        av_frame_free(&fr);
        throw std::runtime_error("cudaStreamSynchronize before NVENC submit");
    }

    fr->pts = d.pts++;
    AV_OK(avcodec_send_frame(d.enc, fr), "avcodec_send_frame");
    // The encoder takes its own reference; release ours.
    av_frame_free(&fr);

    for (;;) {
        int r = avcodec_receive_packet(d.enc, d.pkt);
        if (r == AVERROR(EAGAIN) || r == AVERROR_EOF) break;
        if (r < 0) die("avcodec_receive_packet", r);
        av_packet_rescale_ts(d.pkt, d.enc->time_base, d.stream->time_base);
        d.pkt->stream_index = d.stream->index;
        AV_OK(av_interleaved_write_frame(d.fmt, d.pkt), "av_interleaved_write_frame");
        av_packet_unref(d.pkt);
    }
}

void NvencWriter::close() {
    auto& d = *d_;
    if (!d.open) return;

    // Flush encoder.
    int r = avcodec_send_frame(d.enc, nullptr);
    if (r < 0 && r != AVERROR_EOF) die("flush send_frame", r);
    for (;;) {
        r = avcodec_receive_packet(d.enc, d.pkt);
        if (r == AVERROR(EAGAIN) || r == AVERROR_EOF) break;
        if (r < 0) die("flush receive_packet", r);
        av_packet_rescale_ts(d.pkt, d.enc->time_base, d.stream->time_base);
        d.pkt->stream_index = d.stream->index;
        AV_OK(av_interleaved_write_frame(d.fmt, d.pkt), "flush write_frame");
        av_packet_unref(d.pkt);
    }

    AV_OK(av_write_trailer(d.fmt), "av_write_trailer");
    if (d.fmt && !(d.fmt->oformat->flags & AVFMT_NOFILE))
        avio_closep(&d.fmt->pb);

    av_packet_free(&d.pkt);
    av_buffer_unref(&d.hw_frm_ref);
    av_buffer_unref(&d.hw_dev_ref);
    avcodec_free_context(&d.enc);
    avformat_free_context(d.fmt);
    d.fmt = nullptr;

    d.open = false;
}

} // namespace lidartps
