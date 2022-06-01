#include "noteshrink.h"

#include <cassert>
#include <cmath>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <map>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

namespace
{
int const bitsPerSample = 6;
}

static void SamplePixels(NSHRgb* input, size_t inputSize, NSHOption o, std::vector<NSHRgb>& samples)
{
    samples.clear();
    size_t numSamples = (size_t)std::min(std::max(float(inputSize) * o.SampleFraction, float(0)), float(inputSize));
    size_t interval = std::max((size_t)1, inputSize / numSamples);
    for (size_t i = 0; i < inputSize; i += interval) {
        samples.push_back(input[i]);
    }

    //    std::vector<NSHRgb> shuffled;
    //    for (int i = 0; i < inputSize; i++) {
    //        shuffled.push_back(input[i]);
    //    }
    //    std::random_shuffle(shuffled.begin(), shuffled.end());
    //
    //    for (int i = 0; i < numSamples; i++) {
    //        samples.push_back(shuffled[i]);
    //    }
}


static void Quantize(std::vector<NSHRgb> const& image, int bitsPerChannel, std::vector<uint32_t>& quantized)
{
    uint8_t shift = 8 - bitsPerChannel;
    uint8_t halfbin = uint8_t((1 << shift) >> 1);

    quantized.clear();
    quantized.reserve(image.size());

    for (size_t i = 0; i < image.size(); i++) {
        uint32_t r = ((uint8_t(image[i].R) >> shift) << shift) + halfbin;
        uint32_t g = ((uint8_t(image[i].G) >> shift) << shift) + halfbin;
        uint32_t b = ((uint8_t(image[i].B) >> shift) << shift) + halfbin;
        uint32_t p = (((r << 8) | g) << 8) | b;
        quantized.push_back(p);
    }
}


static void RgbToHsv(NSHRgb p, float& h, float& s, float& v)
{
    float r = p.R / 255.0f;
    float g = p.G / 255.0f;
    float b = p.B / 255.0f;
    float max = std::max(std::max(r, g), b);
    float min = std::min(std::min(r, g), b);
    h = max - min;
    if (h > 0) {
        if (max == r) {
            h = (g - b) / h;
            if (h < 0) {
                h += 6;
            }
        } else if (max == g) {
            h = 2 + (b - r) / h;
        } else {
            h = 4 + (r - g) / h;
        }
    }
    h /= 6;
    s = max - min;
    if (max > 0) {
        s /= max;
    }
    v = max;
}

static NSHRgb HsvToRgb(float h, float s, float v)
{
    float r = v;
    float g = v;
    float b = v;
    if (s > 0) {
        h *= 6.;
        int i = int(h);
        float f = h - float(i);
        switch (i) {
            default:
            case 0:
                g *= 1 - s * (1 - f);
                b *= 1 - s;
                break;
            case 1:
                r *= 1 - s * f;
                b *= 1 - s;
                break;
            case 2:
                r *= 1 - s;
                b *= 1 - s * (1 - f);
                break;
            case 3:
                r *= 1 - s;
                g *= 1 - s * f;
                break;
            case 4:
                r *= 1 - s * (1 - f);
                g *= 1 - s;
                break;
            case 5:
                g *= 1 - s;
                b *= 1 - s * f;
                break;
        }
    }
    NSHRgb p;
    p.R = r * 255;
    p.G = g * 255;
    p.B = b * 255;
    return p;
}



static NSHRgb FindBackgroundColor(std::vector<NSHRgb> const& image, int bitsPerChannel)
{
    std::vector<uint32_t> quantized;
    Quantize(image, bitsPerChannel, quantized);
    std::map<uint32_t, int> count;
    int maxcount = 1;
    uint32_t maxvalue = quantized[0];
    for (size_t i = 1; i < quantized.size(); i++) {
        uint32_t v = quantized[i];
        int c = count[v] + 1;
        if (c > maxcount) {
            maxcount = c;
            maxvalue = v;
        }
        count[v] = c;
    }

    uint8_t shift = 8 - bitsPerChannel;
    uint8_t r = (maxvalue >> 16) & 0xff;
    uint8_t g = (maxvalue >> 8) & 0xff;
    uint8_t b = maxvalue & 0xff;

    NSHRgb bg;
    bg.R = r;
    bg.G = g;
    bg.B = b;

    return bg;
}


static void CreateForegroundMask(NSHRgb bgColor, std::vector<NSHRgb> const& samples, NSHOption option, std::vector<bool>& mask)
{
    float hBg, sBg, vBg;
    RgbToHsv(bgColor, hBg, sBg, vBg);
    std::vector<float> sSamples;
    sSamples.reserve(samples.size());
    std::vector<float> vSamples;
    vSamples.reserve(samples.size());
    for (size_t i = 0; i < samples.size(); i++) {
        float h, s, v;
        RgbToHsv(samples[i], h, s, v);
        sSamples.push_back(s);
        vSamples.push_back(v);
    }

    mask.clear();
    mask.reserve(samples.size());
    for (size_t i = 0; i < samples.size(); i++) {
        float sDiff = fabs(sBg - sSamples[i]);
        float vDiff = fabs(vBg - vSamples[i]);
        bool fg = vDiff >= option.BrightnessThreshold || sDiff >= option.SaturationThreshold;
        mask.push_back(fg);
    }
}


static float SquareDistance(NSHRgb a, NSHRgb b)
{
    float squareDistance = 0;
    squareDistance += (a.R - b.R) * (a.R - b.R);
    squareDistance += (a.G - b.G) * (a.G - b.G);
    squareDistance += (a.B - b.B) * (a.B - b.B);
    return squareDistance;
}


static int Closest(NSHRgb p, std::vector<NSHRgb> const& means)
{
    int idx = 0;
    float minimum = SquareDistance(p, means[0]);
    for (size_t i = 0; i < means.size(); i++) {
        float squaredDistance = SquareDistance(p, means[i]);
        if (squaredDistance < minimum) {
            minimum = squaredDistance;
            idx = i;
        }
    }
    return idx;
}


static NSHRgb Add(NSHRgb a, NSHRgb b)
{
    NSHRgb r;
    r.R = a.R + b.R;
    r.G = a.G + b.G;
    r.B = a.B + b.B;
    return r;
}


static NSHRgb Mul(NSHRgb a, float scalar)
{
    NSHRgb r;
    r.R = a.R * scalar;
    r.G = a.G * scalar;
    r.B = a.B * scalar;
    return r;
}


static void KMeans(std::vector<NSHRgb> const& data, int k, int maxItr, std::vector<NSHRgb>& means)
{
    means.clear();
    means.reserve(k);
    for (int i = 0; i < k; i++) {
        float h = float(i) / float(k - 1);
        NSHRgb p = HsvToRgb(h, 1, 1);
        means.push_back(p);
    }

    std::vector<int> clusters(data.size());
    for (size_t i = 0; i < data.size(); i++) {
        NSHRgb d = data[i];
        clusters[i] = Closest(d, means);
    }

    std::vector<int> mLen(k);
    for (int itr = 0; itr < maxItr; itr++) {
        for (size_t i = 0; i < k; i++) {
            NSHRgb p;
            p.R = p.G = p.B = 0;
            means[i] = p;
            mLen[i] = 0;
        }
        for (size_t i = 0; i < data.size(); i++) {
            NSHRgb p = data[i];
            int cluster = clusters[i];
            NSHRgb m = Add(means[cluster], p);
            means[cluster] = m;
            mLen[cluster] = mLen[cluster] + 1;
        }
        for (size_t i = 0; i < means.size(); i++) {
            int len = std::max(1, mLen[i]);
            NSHRgb m = Mul(means[i], 1 / float(len));
            means[i] = m;
        }
        int changes = 0;
        for (size_t i = 0; i < data.size(); i++) {
            NSHRgb p = data[i];
            int cluster = Closest(p, means);
            if (cluster != clusters[i]) {
                changes++;
                clusters[i] = cluster;
            }
        }
        if (changes == 0) {
            break;
        }
    }
}


static void CreatePalette(std::vector<NSHRgb> const& samples, NSHOption option, std::vector<NSHRgb>& outPalette, NSHRgb& bgColor)
{
    bgColor = FindBackgroundColor(samples, bitsPerSample);

    std::vector<bool> fgMask;
    CreateForegroundMask(bgColor, samples, option, fgMask);

    std::vector<NSHRgb> data;

    for (int i = 0; i < samples.size(); i++) {
        if (!fgMask[i]) {
            continue;
        }
        NSHRgb v = samples[i];
        data.push_back(v);
    }

    std::vector<NSHRgb> means;
    KMeans(samples, outPalette.size() - 1, option.KmeansMaxIter, means);

    size_t idx = 0;
    outPalette[idx++] = bgColor;
    for (size_t i = 0; i < means.size(); i++) {
        NSHRgb c = means[i];
        c.R = round(c.R);
        c.G = round(c.G);
        c.B = round(c.B);
        outPalette[idx++] = c;
    }
}

static void ApplyPalette(std::vector<NSHRgb>& img, std::vector<NSHRgb>& palette, NSHRgb origBgColor, NSHRgb bgColor, NSHOption option, std::vector<NSHRgb>& result)
{
    std::vector<bool> fgMask;
    CreateForegroundMask(origBgColor, img, option, fgMask);
    //    std::vector<NSHRgb> result;
    for (int i = 0; i < img.size(); i++) {
        if (!fgMask[i]) {
            result.push_back(bgColor);
            continue;
        }
        NSHRgb p = img[i];
        int minIdx = Closest(p, palette);
        if (minIdx == 0) {
            result.push_back(bgColor);
        } else {
            result.push_back(palette[minIdx]);
        }
    }
}

static void SaturatePalette(std::vector<NSHRgb>& palette, std::vector<NSHRgb>& resultPalette)
{
      float maxSat = 0;
      float minSat = 1;

      for (int i = 0; i < palette.size(); i++) {
          float h, s, v;
          RgbToHsv(palette[i], h, s, v);
          maxSat = std::max(maxSat, s);
          minSat = std::min(minSat, s);
      }

//      minSat = 1;
//      maxSat = 0;
      for (int i = 0; i < palette.size(); i++) {
          float h, s, v;
          RgbToHsv(palette[i], h, s, v);
          float newSat = (s - minSat) / (maxSat - minSat);
          resultPalette[i] = HsvToRgb(h, newSat, v);
      }
      return;
    std::vector<float> pArray(palette.size() * 3);
    for (int i = 0; i < palette.size(); i++) {
        NSHRgb p = palette[i];
        int j = i * 3;
        pArray[j] = p.R;
        pArray[j + 1] = p.G;
        pArray[j + 2] = p.B;
    }

    float min = *std::min_element(pArray.begin(), pArray.end());
    float max = *std::max_element(pArray.begin(), pArray.end());

//    min = 255;
//    max = 0;
    float diff = max - min;
    for (int i = 0; i < palette.size(); i++) {
        //        NSHRgb p = palette[i];
        //        int j = i * 3;
        //        pArray[j] = p.R;
        //        pArray[j + 1] = p.G;
        //        pArray[j + 2] = p.B;

        resultPalette[i].R = (uint8_t)(255 * (palette[i].R - min) / diff);
        resultPalette[i].G = (uint8_t)(255 * (palette[i].G - min) / diff);
        resultPalette[i].B = (uint8_t)(255 * (palette[i].B - min) / diff);
    }

    /*    for (int i = 0; i < palette.size(); i++) {
            NSHRgb p = palette[i];
            float r = p.R;
            float g = p.G;
            float b = p.B;
            float max = std::max(std::max(r, g), b);
            float min = std::min(std::min(r, g), b);
            if (r > g) {
                if (r > b) {
                    r = 255;
                    if (b < g) {
                        b = 0;
                    } else {
                        g = 0;
                    }
                } else {
                    b = 255;
                    g = 0;
                }
            } else if (g > b) {
                g = 255;
                if (b < r) {
                    b = 0;
                } else {
                    r = 0;
                }
            } else {
                b = 255;
                r = 0;
            }

            p.R = r;
            p.G = g;
            p.B = b;
            resultPalette[i] = p;
        }*/
}
static void SaturatePaletteImg(std::vector<NSHRgb>& palette, std::vector<NSHRgb>& imPalette, std::vector<NSHRgb>& imgResultPalette)
{
    float maxSat = 0;
    float minSat = 1;

    for (int i = 0; i < palette.size(); i++) {
        float h, s, v;
        RgbToHsv(palette[i], h, s, v);
        maxSat = std::max(maxSat, s);
        minSat = std::min(minSat, s);
    }

    for (int i = 0; i < imPalette.size(); i++) {
        float h, s, v;
        RgbToHsv(imPalette[i], h, s, v);
        float newSat = (s - minSat) / (maxSat - minSat);
        imgResultPalette[i] = HsvToRgb(h, newSat, v);
    }
    return;
    std::vector<float> pArray(palette.size() * 3);
    for (int i = 0; i < palette.size(); i++) {
        NSHRgb p = palette[i];
        int j = i * 3;
        pArray[j] = p.R;
        pArray[j + 1] = p.G;
        pArray[j + 2] = p.B;
    }

    float min = *std::min_element(pArray.begin(), pArray.end());
    float max = *std::max_element(pArray.begin(), pArray.end());
    float diff = max - min;
    for (int i = 0; i < imPalette.size(); i++) {
        //        NSHRgb p = palette[i];
        //        int j = i * 3;
        //        pArray[j] = p.R;
        //        pArray[j + 1] = p.G;
        //        pArray[j + 2] = p.B;

        imgResultPalette[i].R = (uint8_t)(255 * (imPalette[i].R - min) / diff);
        imgResultPalette[i].G = (uint8_t)(255 * (imPalette[i].G - min) / diff);
        imgResultPalette[i].B = (uint8_t)(255 * (imPalette[i].B - min) / diff);
    }

    /*    for (int i = 0; i < palette.size(); i++) {
            NSHRgb p = palette[i];
            float r = p.R;
            float g = p.G;
            float b = p.B;
            float max = std::max(std::max(r, g), b);
            float min = std::min(std::min(r, g), b);
            if (r > g) {
                if (r > b) {
                    r = 255;
                    if (b < g) {
                        b = 0;
                    } else {
                        g = 0;
                    }
                } else {
                    b = 255;
                    g = 0;
                }
            } else if (g > b) {
                g = 255;
                if (b < r) {
                    b = 0;
                } else {
                    r = 0;
                }
            } else {
                b = 255;
                r = 0;
            }

            p.R = r;
            p.G = g;
            p.B = b;
            resultPalette[i] = p;
        }*/
}


static float vec3bDist(cv::Vec3b a, cv::Vec3b b)
{
    return sqrt(pow((float)a[0] - b[0], 2) + pow((float)a[1] - b[1], 2) + pow((float)a[2] - b[2], 2));
}

static cv::Vec3b findClosestPaletteColor(cv::Vec3b color, std::vector<NSHRgb>& palette)
{
    //    int i = 0;
    int minI = 0;
    //    cv::Vec3b diff = color - palette.at<cv::Vec3b>(0);

    cv::Vec3b Vec3bAt0 = cv::Vec3b(palette[0].B, palette[0].G, palette[0].R);
    float minDistance = vec3bDist(color, Vec3bAt0);
    for (int i = 0; i < palette.size(); ++i) {
        float distance = vec3bDist(color, cv::Vec3b(palette[i].B, palette[i].G, palette[i].R));
        if (distance < minDistance) {
            minDistance = distance;
            minI = i;
        }
    }
    return cv::Vec3b(palette[minI].B, palette[minI].G, palette[minI].R);
}

static cv::Mat floydSteinberg(cv::Mat imgOrig, std::vector<NSHRgb>& palette)
{
    cv::Mat img = imgOrig.clone();
    cv::Mat resImg = img.clone();
    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            cv::Vec3b newpixel = findClosestPaletteColor(img.at<cv::Vec3b>(i, j), palette);
            resImg.at<cv::Vec3b>(i, j) = newpixel;

            /* for (int k = 0; k < 3; ++k) {
                 int quant_error = (int)img.at<cv::Vec3b>(i, j)[k] - newpixel[k];
                 if (i + 1 < img.rows)
                     img.at<cv::Vec3b>(i + 1, j)[k] = std::min(255, std::max(0, (int)img.at<cv::Vec3b>(i + 1, j)[k] + (7 * quant_error) / 16));
                 if (i - 1 > 0 && j + 1 < img.cols)
                     img.at<cv::Vec3b>(i - 1, j + 1)[k] = std::min(255, std::max(0, (int)img.at<cv::Vec3b>(i - 1, j + 1)[k] + (3 * quant_error) / 16));
                 if (j + 1 < img.cols)
                     img.at<cv::Vec3b>(i, j + 1)[k] = std::min(255, std::max(0, (int)img.at<cv::Vec3b>(i, j + 1)[k] + (5 * quant_error) / 16));
                 if (i + 1 < img.rows && j + 1 < img.cols)
                     img.at<cv::Vec3b>(i + 1, j + 1)[k] = std::min(255, std::max(0, (int)img.at<cv::Vec3b>(i + 1, j + 1)[k] + (1 * quant_error) / 16));
             }*/
            for (int k = 0; k < 3; k++) {
                int quant_error = (int)img.at<cv::Vec3b>(i, j)[k] - newpixel[k];
                if (j + 1 < img.cols)
                    img.at<cv::Vec3b>(i, j + 1)[k] = fmin(255, fmax(0, (int)img.at<cv::Vec3b>(i, j + 1)[k] + (7 * quant_error) / 16));
                if (i + 1 < img.rows && j - 1 >= 0)
                    img.at<cv::Vec3b>(i + 1, j - 1)[k] = fmin(255, fmax(0, (int)img.at<cv::Vec3b>(i + 1, j - 1)[k] + (3 * quant_error) / 16));
                if (i + 1 < img.rows)
                    img.at<cv::Vec3b>(i + 1, j)[k] = fmin(255, fmax(0, (int)img.at<cv::Vec3b>(i + 1, j)[k] + (5 * quant_error) / 16));
                if (i + 1 < img.rows && j + 1 < img.cols)
                    img.at<cv::Vec3b>(i + 1, j + 1)[k] = fmin(255, fmax(0, (int)img.at<cv::Vec3b>(i + 1, j + 1)[k] + (1 * quant_error) / 16));
            }
        }
    }
    return resImg;
}



extern "C" NSHOption NSHMakeDefaultOption()
{
    NSHOption o;
    o.SampleFraction = 0.05f;
    o.BrightnessThreshold = 0.25f;
    o.SaturationThreshold = 0.20f;
    o.KmeansMaxIter = 40;
    o.Saturate = true;
    o.WhiteBackground = false;
    return o;
}


extern "C" bool NSHCreatePalette(std::vector<NSHRgb>& input, size_t inputSize, NSHOption option, NSHRgb* palette, size_t paletteSize, std::vector<NSHRgb>& result, int width, int height)
{
    if (input.empty() || !palette) {
        return false;
    }
    if (paletteSize < 2) {
        return false;
    }

    std::vector<NSHRgb> samples;
    SamplePixels(input.data(), inputSize, option, samples);
    NSHRgb origBgColor;
    std::vector<NSHRgb> pal(paletteSize);
    std::vector<NSHRgb> resultPal(paletteSize);
    CreatePalette(samples, option, pal, origBgColor);
    NSHRgb bgColor = origBgColor;
    if (1 == 2 && option.Saturate) {
        SaturatePalette(pal, resultPal);
    } else {


                SaturatePalette(pal, resultPal);
        // after saturation
      /*  pal[0].R = 255;
        pal[0].G = 255;
        pal[0].B = 183;
        pal[1].R = 14;
        pal[1].G = 16;
        pal[1].B = 10;
        pal[2].R = 235;
        pal[2].G = 52;
        pal[2].B = 47;
        pal[3].R = 29;
        pal[3].G = 129;
        pal[3].B = 98;
        pal[4].R = 83;
        pal[4].G = 85;
        pal[4].B = 64;
        pal[5].R = 2;
        pal[5].G = 4;
        pal[5].B = 0;
        pal[6].R = 41;
        pal[6].G = 44;
        pal[6].B = 26;
        pal[7].R = 134;
        pal[7].G = 135;
        pal[7].B = 107;*/
//        for (int i = 0; i < paletteSize; i++) {
//            resultPal[i] = pal[i];
//        }
    }
    //    bgColor = origBgColor = resultPal[0];
    if (/*1==1||*/ option.WhiteBackground) {
        bgColor = NSHRgb{ 255, 255, 255 };
    }

    std::vector<NSHRgb> paletteAppliedResultimg;
    ApplyPalette(input, resultPal, origBgColor, bgColor, option, result);
    // SaturatePaletteImg(resultPal,paletteAppliedResultimg, result);
    uchar dataTest[] = { 255, 255, 183, 130, 133, 105, 16, 17, 11, 85, 88, 65, 44, 45, 29, 4, 4, 0, 31, 131, 102, 235, 59, 51 };
/*    pal[0].R = 255;
    pal[0].G = 255;
    pal[0].B = 183;
    pal[1].R = 14;
    pal[1].G = 16;
    pal[1].B = 10;
    pal[2].R = 235;
    pal[2].G = 52;
    pal[2].B = 47;
    pal[3].R = 29;
    pal[3].G = 129;
    pal[3].B = 98;
    pal[4].R = 83;
    pal[4].G = 85;
    pal[4].B = 64;
    pal[5].R = 2;
    pal[5].G = 4;
    pal[5].B = 0;
    pal[6].R = 41;
    pal[6].G = 44;
    pal[6].B = 26;
    pal[7].R = 134;
    pal[7].G = 135;
    pal[7].B = 107;
    for (int i = 0; i < paletteSize; i++) {
        resultPal[i] = pal[i];
    }*/

    cv::Mat inputImg = cv::Mat(height, width, CV_8UC3);
    //    for(int i=0;i<height;i++){
    //        for(int j=0;j<width;j++){
    //            mat.at<Vec3b>(i,j);
    //        }
    //    }
    int z = 0;
    for (int y = 0; y < inputImg.rows; y++) {
        for (int x = 0; x < inputImg.cols; x++) {
            // size_t idx = (height - y - 1) * width * 3 + x * 3;
            NSHRgb p = result[z++];
            //            data[idx] = (uint8_t)p.R;
            //            data[idx + 1] = (uint8_t)p.G;
            //            data[idx + 2] = (uint8_t)p.B;

            // get pixel
            cv::Vec3b& color = inputImg.at<cv::Vec3b>(y, x);

            // ... do something to the color ....
            color[0] = p.B;
            color[1] = p.G;
            color[2] = p.R;

            // set pixel
            // image.at<Vec3b>(Point(x,y)) = color;
            // if you copy value
        }
    }

    cv::imshow("Input Image", inputImg);
    //    cv::waitKey(20);
    // cv::Mat img;
    // cv::cvtColor(inputImg, img, cv::COLOR_BGR2Lab);
    cv::Mat floydSteinbergResultImage = floydSteinberg(inputImg, resultPal);
    // cv::cvtColor(floydSteinbergResultImage, img, cv::COLOR_Lab2BGR);
    cv::imshow("floydSteinberg Image", floydSteinbergResultImage);
    cv::waitKey();
    return true;

    // int dim(256);
    cv::Mat lut(1, 256, CV_8UC(1));
    for (int i = 0; i < 256; i++) {
        lut.at<uchar>(i) = dataTest[i % 24]; // first channel  (B)
        // lut.at<int>(i)= pal[i%8].R;   // first channel  (B)
        // lut.at<cv::Vec3b>(i)[1]= pal[i%8].G; // second channel (G)
        // lut.at<cv::Vec3b>(i)[2]= pal[i%8].B; // ...            (R)
    }


    // cv::Mat pleteImg(8);
    //    cv::Mat mat = cv::Mat(8, 1, CV_8UC3);
    //    uchar temp[8][3];
    //    for(int i=0;i<8;i++){
    //        //mat.at
    //            temp[i][0] = resultPal[i].R;
    //            temp[i][1] = resultPal[i].G;
    //            temp[i][2] = resultPal[i].B;
    //    }
    //    mat.data = temp[0];


    cv::Mat resultImage;
    cv::LUT(inputImg, lut, resultImage);

    cv::imshow("Result Image", resultImage);


    // ApplyPalette(img, resultPal, origBgColor, bgColor, option, result);
    return true;
}



//        original palette
/* pal[0].R = 230;
 pal[0].G = 230;
 pal[0].B = 182;

pal[1].R = 141;
pal[1].G = 142;
pal[1].B = 124;

pal[2].R = 213;
pal[2].G = 76;
pal[2].B = 77;

pal[3].R = 98;
pal[3].G = 99;
pal[3].B = 87;

pal[4].R = 72;
pal[4].G = 73;
pal[4].B = 67;

pal[5].R = 61;
pal[5].G = 62;
pal[5].B = 60;

pal[6].R = 82;
pal[6].G = 148;
pal[6].B = 126;

pal[7].R = 224;
pal[7].G = 129;
pal[7].B = 117;*/

// after saturation
/*        pal[0].R = 255;
        pal[0].G = 255;
        pal[0].B = 183;
        pal[1].R = 14;
        pal[1].G = 16;
        pal[1].B = 10;
        pal[2].R = 235;
        pal[2].G = 52;
        pal[2].B = 47;
        pal[3].R = 29;
        pal[3].G = 129;
        pal[3].B = 98;
        pal[4].R = 83;
        pal[4].G = 85;
        pal[4].B = 64;
        pal[5].R = 2;
        pal[5].G = 4;
        pal[5].B = 0;
        pal[6].R = 41;
        pal[6].G = 44;
        pal[6].B = 26;
        pal[7].R = 134;
        pal[7].G = 135;
        pal[7].B = 107;*/