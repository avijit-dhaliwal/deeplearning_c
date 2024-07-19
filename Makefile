CC = gcc
CFLAGS = -I./include -Wall -Wextra -pedantic -std=c11 -O2
LDFLAGS = -lm -lcurl -lz

SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin
INCLUDE_DIR = include
TEST_DIR = tests

SOURCES = $(wildcard $(SRC_DIR)/*.c)
OBJECTS = $(SOURCES:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)
DEPS = $(OBJECTS:.o=.d)

LIB_NAME = libdeeplearning.a
LIB_PATH = $(BIN_DIR)/$(LIB_NAME)

EXAMPLE_SOURCES = $(wildcard examples/*.c)
EXAMPLE_EXECUTABLES = $(EXAMPLE_SOURCES:examples/%.c=$(BIN_DIR)/%)

TEST_SOURCES = $(wildcard $(TEST_DIR)/*.c)
TEST_EXECUTABLES = $(TEST_SOURCES:$(TEST_DIR)/%.c=$(BIN_DIR)/%)

.PHONY: all clean examples tests

all: $(LIB_PATH) examples tests

$(LIB_PATH): $(OBJECTS)
    @mkdir -p $(BIN_DIR)
    ar rcs $@ $^

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
    @mkdir -p $(OBJ_DIR)
    $(CC) $(CFLAGS) -MMD -MP -c $< -o $@

examples: $(EXAMPLE_EXECUTABLES)

$(BIN_DIR)/%: examples/%.c $(LIB_PATH)
    @mkdir -p $(BIN_DIR)
    $(CC) $(CFLAGS) $< $(LIB_PATH) $(LDFLAGS) -o $@

tests: $(TEST_EXECUTABLES)

$(BIN_DIR)/%: $(TEST_DIR)/%.c $(LIB_PATH)
    @mkdir -p $(BIN_DIR)
    $(CC) $(CFLAGS) $< $(LIB_PATH) $(LDFLAGS) -o $@

clean:
    rm -rf $(OBJ_DIR) $(BIN_DIR)

-include $(DEPS)