
macro(begin_new_project name)

	project(${name})
	add_definitions(-DCVAPI_EXPORTS)

	include_directories("${CMAKE_CURRENT_SOURCE_DIR}"
	                    "${CMAKE_CURRENT_BINARY_DIR}"
	                    "${CMAKE_BINARY_DIR}"
						"${OpenCV_INCLUDE_DIRS}")

	file(GLOB files_srcs "*.cpp")
	file(GLOB files_int_hdrs "*.h*")
	source_group("Src" FILES ${files_srcs})
	source_group("Include" FILES ${files_int_hdrs})

	set(the_target "${name}")

	set(file_list_ ${files_srcs} ${files_int_hdrs})
endmacro()

macro(end_new_project name)

	# For dynamic link numbering convenions
	set_target_properties(${the_target} PROPERTIES
	    VERSION ${DETECT_VERSION}
	    SOVERSION ${DETECT_VERSION}
	    OUTPUT_NAME "${the_target}${DETECT_VERSION}"
	    )
	# Additional target properties
	set_target_properties(${the_target} PROPERTIES
	    DEBUG_POSTFIX "${DETECT_DEBUG_POSTFIX}"
	    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/"
	    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/"
	    INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/lib"
	    )

	if(MSVC)
	    if(CMAKE_CROSSCOMPILING)
	        set_target_properties(${the_target} PROPERTIES
	            LINK_FLAGS "/NODEFAULTLIB:secchk"
	            )
	    endif()
	    set_target_properties(${the_target} PROPERTIES
	        LINK_FLAGS "/NODEFAULTLIB:libc;libcmt"
	        )
	endif()

	# Dependencies of this target:
	if(ARGN)
		add_dependencies(${the_target} ${ARGN})
	endif()
	
	# Add the required libraries for linking:
	target_link_libraries(${the_target} ${DETECT_LINKER_LIBS} ${QT_LIBRARIES} ${ARGN})

endmacro()


macro(new_library name)

	begin_new_project(${name})
	
	add_library(${the_target} ${LIB_TYPE} ${file_list_})
	#SET_TARGET_PROPERTIES (${the_target} PROPERTIES DEFINE_SYMBOL "SFM_API_EXPORTS")
	
	end_new_project(${name} ${ARGN})
endmacro()


macro(new_executable name)

	begin_new_project(${name})
	
	add_executable(${the_target} ${file_list_})
	
	end_new_project(${name} ${ARGN})
	
endmacro()
