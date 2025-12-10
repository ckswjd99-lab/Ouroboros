// custom_h264.c  ─  한 프레임 H.264 인코드→디코드 후 모션-벡터 추출
#include <stdint.h>
#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
#include <libavutil/motion_vector.h>
#include <libswscale/swscale.h>
#include <string.h>

typedef struct {
    int16_t dst_x, dst_y;        // 블록 좌표 (좌상단)
    int32_t motion_x, motion_y;  // quarter-pel
    int32_t motion_scale;        // 보통 1 or 2
} MVInfo;

static AVCodecContext *enc = NULL, *dec = NULL;
static AVFrame *enc_fr = NULL, *dec_fr = NULL;
static struct SwsContext *sws = NULL;
static AVPacket *pkt = NULL;
static int pts = 0;

// 원본 코드에서 사용한 h264 파라미터를 default로 사용
static const char *DEFAULT_X264_PARAMS =
    "keyint=9999:min-keyint=9999:no-scenecut=1:bframes=0:"
    "ref=1:partitions=i16x16,p16x16:8x8dct=0:rc-lookahead=0";

int mv_init(int w, int h, int fps, const char* x264_params)
{
    const AVCodec *enc_c = avcodec_find_encoder_by_name("libx264");
    const AVCodec *dec_c = avcodec_find_decoder(AV_CODEC_ID_H264);
    if (!enc_c || !dec_c) return -1;

    enc = avcodec_alloc_context3(enc_c);
    dec = avcodec_alloc_context3(dec_c);
    if (!enc || !dec) return -1;

    enc->width = w;  enc->height = h;
    enc->pix_fmt = AV_PIX_FMT_YUV420P;
    enc->time_base = (AVRational){1,fps};
    enc->gop_size = 9999; enc->max_b_frames = 0;
    av_opt_set(enc->priv_data,"preset","fast",0);
    av_opt_set(enc->priv_data,"profile","baseline",0);
    av_opt_set(enc->priv_data,"tune","zerolatency",0);
    if (x264_params && strlen(x264_params) > 0)
        av_opt_set(enc->priv_data,"x264-params", x264_params, 0);
    else
        av_opt_set(enc->priv_data,"x264-params", DEFAULT_X264_PARAMS, 0);

    if (avcodec_open2(enc,enc_c,NULL)<0) return -1;

    av_opt_set_int(dec,"flags2",AV_CODEC_FLAG2_EXPORT_MVS,0);
    if (avcodec_open2(dec,dec_c,NULL)<0) return -1;

    enc_fr=av_frame_alloc(); dec_fr=av_frame_alloc(); pkt=av_packet_alloc();
    if(!enc_fr||!dec_fr||!pkt) return -1;
    enc_fr->format=AV_PIX_FMT_YUV420P; enc_fr->width=w; enc_fr->height=h;
    if(av_frame_get_buffer(enc_fr,32)<0) return -1;

    sws = sws_getContext(w,h,AV_PIX_FMT_BGR24,w,h,AV_PIX_FMT_YUV420P,
                         SWS_BILINEAR,NULL,NULL,NULL);
    return sws?0:-1;
}

int mv_process(uint8_t* bgr, MVInfo *out, int max_out)
{
    int w=enc->width,h=enc->height;
    uint8_t* in[1]={bgr}; int ls[1]={3*w};
    sws_scale(sws,(const uint8_t* const*)in,ls,0,h,enc_fr->data,enc_fr->linesize);

    enc_fr->pts=pts++;
    if(avcodec_send_frame(enc,enc_fr)<0) return -1;

    while(avcodec_receive_packet(enc,pkt)==0){
        if(avcodec_send_packet(dec,pkt)<0){ av_packet_unref(pkt); return -1;}
        av_packet_unref(pkt);

        if(avcodec_receive_frame(dec,dec_fr)==0){
            AVFrameSideData *sd=av_frame_get_side_data(dec_fr,
                                        AV_FRAME_DATA_MOTION_VECTORS);
            int cnt=0;
            if(sd){
                const AVMotionVector *mv=(const AVMotionVector*)sd->data;
                int tot=sd->size/sizeof(*mv);
                for(int i=0;i<tot&&cnt<max_out;i++){
                    out[cnt].dst_x     = mv[i].dst_x;
                    out[cnt].dst_y     = mv[i].dst_y;
                    out[cnt].motion_x  = mv[i].motion_x;
                    out[cnt].motion_y  = mv[i].motion_y;
                    out[cnt].motion_scale = mv[i].motion_scale;
                    ++cnt;
                }
            }
            av_frame_unref(dec_fr);
            return cnt;
        }
    }
    return 0;
}

// 새 시그니처: encoded_size_out, mv_cnt_out 포인터 추가
int mv_process_and_encode(uint8_t* bgr,
                          MVInfo *out, int max_out,
                          uint8_t* out_buf, int out_buf_size,
                          int *encoded_size_out, int *mv_cnt_out)
{
    int w=enc->width, h=enc->height;
    uint8_t* in[1]={bgr}; int ls[1]={3*w};
    sws_scale(sws,(const uint8_t* const*)in,ls,0,h,enc_fr->data,enc_fr->linesize);

    enc_fr->pts=pts++;
    if(avcodec_send_frame(enc,enc_fr)<0) return -1;

    int encoded_size = 0;
    int mv_cnt = 0;
    while(avcodec_receive_packet(enc,pkt)==0){
        if(encoded_size + pkt->size <= out_buf_size) {
            memcpy(out_buf + encoded_size, pkt->data, pkt->size); // 이어붙이기
            encoded_size += pkt->size;
        } else {
            av_packet_unref(pkt);
            return -1; // 버퍼 초과
        }
        if(avcodec_send_packet(dec,pkt)<0){ av_packet_unref(pkt); return -1;}
        av_packet_unref(pkt);

        if(avcodec_receive_frame(dec,dec_fr)==0){
            AVFrameSideData *sd=av_frame_get_side_data(dec_fr, AV_FRAME_DATA_MOTION_VECTORS);
            int cnt=0;
            if(sd){
                const AVMotionVector *mv=(const AVMotionVector*)sd->data;
                int tot=sd->size/sizeof(*mv);
                for(int i=0;i<tot&&cnt<max_out;i++){
                    out[cnt].dst_x     = mv[i].dst_x;
                    out[cnt].dst_y     = mv[i].dst_y;
                    out[cnt].motion_x  = mv[i].motion_x;
                    out[cnt].motion_y  = mv[i].motion_y;
                    out[cnt].motion_scale = mv[i].motion_scale;
                    ++cnt;
                }
            }
            av_frame_unref(dec_fr);
            mv_cnt = cnt;
        }
    }
    *encoded_size_out = encoded_size;
    *mv_cnt_out = mv_cnt;
    return 0;
}

void mv_close(void){
    avcodec_free_context(&enc); avcodec_free_context(&dec);
    av_frame_free(&enc_fr); av_frame_free(&dec_fr);
    av_packet_free(&pkt); if(sws) sws_freeContext(sws);
}
