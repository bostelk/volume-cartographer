//
// Created by Seth Parker on 10/20/15.
//

#ifndef VC_TEXTURE_H
#define VC_TEXTURE_H

#include <opencv2/opencv.hpp>

#include "../vc_defines.h"
#include "UVMap.h"

namespace volcart {
    class Texture {
    public:
        Texture(){};
        Texture(int width, int height) { _width = width; _height = height; };

        // Get metadata
        int    width()  { return _width; };
        int    height() { return _height; };
        size_t numberOfImages()   { return _images.size(); };
        bool   hasImages() { return _images.size() > 0; };
        bool   hasMap()    { return _uvMap.size()  > 0; };

        // Get/Set UV Map
        volcart::UVMap& uvMap(){ return _uvMap; };
        void uvMap(volcart::UVMap uvMap) { _uvMap = uvMap; };

        // Get/Add Texture Image
        cv::Mat getImage(int id) { return _images[id]; };
        void addImage(cv::Mat image) {
            if ( _images.empty() ) {
                _width = image.cols;
                _height = image.rows;
            }
            _images.push_back(image);
        };

        // Return the intensity for a Point ID
        double intensity( int point_ID, int image_ID = 0 ) {
            cv::Vec2d mapping = _uvMap.get(point_ID);
            if ( mapping != VC_UVMAP_NULL_MAPPING ) {
                int u =  cvRound(mapping[0] * _width);
                int v =  cvRound(mapping[1] * _height);
                return _images[image_ID].at< unsigned short > ( v, u );
            } else {
                return VC_TEXTURE_NO_VALUE;
            }
        }

    private:
        int _width, _height;
        std::vector<cv::Mat> _images;
        volcart::UVMap _uvMap;
    };
} // volcart

#endif //VC_TEXTURE_H
