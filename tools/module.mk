# Every variable in subdir must be prefixed with subdir (emulating a namespace)

TOOLS = \
	tools/memset

TOOLS_SRCS = $(addsuffix .cpp, $(TOOLS))

TOOLS_INCLUDES = $(COMMON_INCLUDES) -I$(TT_METAL_HOME)/tools
TOOLS_LDFLAGS = $(LDFLAGS) -L$(TT_METAL_HOME) -lllrt -ldevice -lcommon -lyaml-cpp -lstdc++fs

TOOLS_DEPS = $(addprefix $(OBJDIR)/, $(TOOLS_SRCS:.cpp=.d))

-include $(TOOLS_DEPS)

# Each module has a top level target as the entrypoint which must match the subdir name
tools: $(OBJDIR)/tools/memset tools/profiler tools/tt_gdb

.PRECIOUS: $(OBJDIR)/tools/%
$(OBJDIR)/tools/memset: $(OBJDIR)/tools/memset.o
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(TOOLS_INCLUDES) -o $@ $^ $(TOOLS_LDFLAGS)

$(OBJDIR)/tools/memset.o: tools/memset.cpp
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(TOOLS_INCLUDES) -c -o $@ $<

include $(TT_METAL_HOME)/tools/tt_gdb/module.mk # needs to compiled after llrt and tt_metal
include $(TT_METAL_HOME)/tools/profiler/module.mk
