# Binaries branch name: ffmpeg/4.x_20231225
# Binaries were created for OpenCV: 62f1a7410d5e5e03d6cee5c95549bf61d5ee98db
ocv_update(FFMPEG_BINARIES_COMMIT "fbac408a47977ee4265f39e7659d33f1dfef5216")
ocv_update(FFMPEG_FILE_HASH_BIN32 "9b755ecbbade0a5b78332e9b4ef2dd1b")
ocv_update(FFMPEG_FILE_HASH_BIN64 "cb4db51ee9a423e6168b9d08bee61efc")
ocv_update(FFMPEG_FILE_HASH_CMAKE "8862c87496e2e8c375965e1277dee1c7")

function(download_win_ffmpeg script_var)
  set(${script_var} "" PARENT_SCOPE)

  set(ids BIN32 BIN64 CMAKE)
  set(name_BIN32 "opencv_videoio_ffmpeg.dll")
  set(name_BIN64 "opencv_videoio_ffmpeg_64.dll")
  set(name_CMAKE "ffmpeg_version.cmake")

  set(FFMPEG_DOWNLOAD_DIR "${OpenCV_BINARY_DIR}/3rdparty/ffmpeg")

  set(status TRUE)
  foreach(id ${ids})
    ocv_download(FILENAME ${name_${id}}
               HASH ${FFMPEG_FILE_HASH_${id}}
               URL
                 "$ENV{OPENCV_FFMPEG_URL}"
                 "${OPENCV_FFMPEG_URL}"
                 "https://raw.githubusercontent.com/opencv/opencv_3rdparty/${FFMPEG_BINARIES_COMMIT}/ffmpeg/"
               DESTINATION_DIR ${FFMPEG_DOWNLOAD_DIR}
               ID FFMPEG
               RELATIVE_URL
               STATUS res)
    if(NOT res)
      set(status FALSE)
    endif()
  endforeach()
  if(status)
    set(${script_var} "${FFMPEG_DOWNLOAD_DIR}/ffmpeg_version.cmake" PARENT_SCOPE)
  endif()
endfunction()

if(OPENCV_INSTALL_FFMPEG_DOWNLOAD_SCRIPT)
  configure_file("${CMAKE_CURRENT_LIST_DIR}/ffmpeg-download.ps1.in" "${CMAKE_BINARY_DIR}/win-install/ffmpeg-download.ps1" @ONLY)
  install(FILES "${CMAKE_BINARY_DIR}/win-install/ffmpeg-download.ps1" DESTINATION "." COMPONENT libs)
endif()

ocv_install_3rdparty_licenses(ffmpeg license.txt readme.txt)
