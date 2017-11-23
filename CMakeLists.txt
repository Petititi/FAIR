cmake_minimum_required ( VERSION 2.6 )


if( CMAKE_CACHE_MAJOR_VERSION>=2 AND CMAKE_CACHE_MINOR_VERSION>6 )
	cmake_policy( SET CMP0015 OLD )
endif( )
cmake_policy( SET CMP0003 NEW )

project ( binarisationPAMI )

if( MSVC )
    set( CMAKE_USE_RELATIVE_PATHS ON CACHE INTERNAL "" FORCE )
endif( )


# ----------------------------------------------------------------------------
#  Current version number:
# ----------------------------------------------------------------------------
set( DETECT_VERSION "0.0.1" )
set( DETECT_DEBUG_POSTFIX d )

string( REGEX MATCHALL "[ 0-9 ]" DETECT_VERSION_PARTS "${DETECT_VERSION}" )

list( GET DETECT_VERSION_PARTS 0 DETECT_VERSION_MAJOR )
list( GET DETECT_VERSION_PARTS 1 DETECT_VERSION_MINOR )
list( GET DETECT_VERSION_PARTS 2 DETECT_VERSION_PATCH )

include( ProjectAdd.cmake REQUIRED )


SET( Boost_ADDITIONAL_VERSIONS "1.43" "1.43.0" "1.44" "1.44.0" "1.45" "1.45.0" "1.46.1" "1.46.0" "1.46" "1.47" "1.48" "1.49" "1.49.0" )
find_path( BOOST_ROOT "boost" 
	DOC "The path to Boost root directory ( before boost_1_XX_X )" )

find_package( Boost COMPONENTS ${BOOST_needed_list} )
if( Boost_FOUND )
	LINK_DIRECTORIES( ${Boost_LIBRARY_DIRS} )
	include_directories( ${Boost_INCLUDE_DIR} )
	message( STATUS " ** Boost library was found, Don't take into account the 'Boost_DIR not found' /!\\" )
endif( )

FIND_PACKAGE( OpenCV REQUIRED )# opencv_calib3d opencv_core opencv_objdetect opencv_features2d opencv_highgui opencv_imgproc opencv_flann opencv_nonfree
set ( OPENCV_CONCAT "${OpenCV_VERSION_MAJOR}${OpenCV_VERSION_MINOR}${OpenCV_VERSION_PATCH}" )

#set( DETECT_LINKER_LIBS "debug;opencv_features2d${OPENCV_CONCAT}d;optimized;opencv_features2d${OPENCV_CONCAT};debug;opencv_highgui${OPENCV_CONCAT}d;optimized;opencv_highgui${OPENCV_CONCAT};" )
set( DETECT_LINKER_LIBS ${OpenCV_LIBS} )
#MESSAGE(STATUS "OpenCV_LIBS : ${DETECT_LINKER_LIBS}")


add_subdirectory( binAlgo )

add_subdirectory( demonstration )

add_subdirectory( F_Mesure )

