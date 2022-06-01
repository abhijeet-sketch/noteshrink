#include "noteshrink.h"
#include <stb/stb_image.h>
#include <stb/stb_image_write.h>
#include <string>
//#include <tic.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
//#include <opencv4.2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc.hpp>

int main()
{
//    cv::Mat image = cv::imread("path");
//    cv::Mat destination = cv::Mat(image.rows, image.cols, image.type());
//    // cvtColor(src, src, CV_GRAY2RGB);
//    GaussianBlur(image, destination, cv::Size(0, 0), 3);
//    addWeighted(image, 1.1, image, 0, 3, image);
//        cv::imshow("destination", image);
//        cv::waitKey();
//        return  0;
  /*  cv::Mat image = cv::imread("path");
//    cv::imshow("input", image);
//    cv::waitKey(50);
    cv::Mat hsv;
    cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    cv::Mat channels[3];
    split(hsv, channels);
    cv::Mat H = channels[0];
    H.convertTo(H, CV_32F);
    cv::Mat S = channels[1];
    S.convertTo(S, CV_32F);
    cv::Mat V = channels[2];
    V.convertTo(V, CV_32F);
    for (int i = 0; i < S.size().height; i++) {
        for (int j = 0; j < S.size().width; j++) {
           *//* H.at<float>(i, j) *= 15;
            if (H.at<float>(i, j) > 255)
                H.at<float>(i, j) = 255;
*//*
            // scale pixel values up or down for channel 1(Saturation)
           S.at<float>(i, j) *= 1.5;
            if (S.at<float>(i, j) > 255)
                S.at<float>(i, j) = 255;

            // scale pixel values up or down for channel 2(Value)
         *//*   V.at<float>(i, j) *= 1.5;
            if (V.at<float>(i, j) > 255)
                V.at<float>(i, j) = 255;*//*
        }
    }
    H.convertTo(H, CV_8U);
    S.convertTo(S, CV_8U);
    V.convertTo(V, CV_8U);
    std::vector<cv::Mat> hsvChannels{ H, S, V };
    cv::Mat hsvNew;
    merge(hsvChannels, hsvNew);
    cv::Mat resultImg;
    cvtColor(hsvNew, resultImg, cv::COLOR_HSV2BGR);

    hsvNew.release();
    H.release();
    S.release();
    V.release();
    hsv.release();

    cv::imshow("result", resultImg);

    cv::waitKey();
    return 0;*/

    std::string file("path");

    int width, height, bpp;
    stbi_uc* pixels = stbi_load(file.c_str(), &width, &height, &bpp, STBI_rgb_alpha);
    std::vector<NSHRgb> img(width * height);

    size_t i = 0;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            size_t idx = (height - y - 1) * width * 4 + x * 4;
            NSHRgb p;
            p.R = pixels[idx];
            p.G = pixels[idx + 1];
            p.B = pixels[idx + 2];
            img[i++] = p;
        }
    }
    free(pixels);

    // cv::Mat edges;
    NSHOption o = NSHMakeDefaultOption();
    std::vector<NSHRgb> palette(6);
    std::vector<NSHRgb> result;
    NSHCreatePalette(img, img.size(), o, palette.data(), palette.size(), result, width, height);
    std::vector<NSHRgb> temVector;
    //    img.swap(temVector);

    //    cv::waitKey();
    int numberOfChannels = 3;
    // unsigned char data[width * height * numberOfChannels];
    // unsigned char *gray_img = malloc(numberOfChannels);

    uint8_t* data = new uint8_t[width * height * numberOfChannels];
    // int* numberArray = new int[n];
    // uint8_t data[width * height * numberOfChannels];

    //    stbi_uc* data[width * height * numberOfChannels];


    i = 0;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            size_t idx = (height - y - 1) * width * numberOfChannels + x * numberOfChannels;
            NSHRgb p = result[i++];
            data[idx] = (uint8_t)p.R;
            data[idx + 1] = (uint8_t)p.G;
            data[idx + 2] = (uint8_t)p.B;
            // data[idx + 3] = 255;
        }
    }
    printf("done");
    //    stbi_write_png("path", width, height, numberOfChannels, data, width * numberOfChannels);
    // stbi_write_jpg("path", width, height, numberOfChannels, data, 70);
    stbi_write_jpg("path", width, height, numberOfChannels, data, width * numberOfChannels);
    return 0;
}