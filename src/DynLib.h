/*
 * Loading a system library at runtime.
 * SPDX-License-Identifier: MIT OR Apache-2.0
 */
#pragma once
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#include <initializer_list>

class DynLib
{
public:
    // The first of `names` that loads. Never unloaded: the GPU objects made
    // through it live as long as the process. On Windows only from System32,
    // where the drivers install these: a bare name would otherwise be looked
    // for in the application's and the current directory first, where a
    // machine without the driver could be handed a planted copy.
    bool open(std::initializer_list<const char*> names) {
        for (auto n : names) {
#ifdef _WIN32
            h_ = (void*)LoadLibraryExA(n, nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
#else
            h_ = dlopen(n, RTLD_NOW | RTLD_LOCAL);
#endif
            if (h_)
                return true;
        }
        return false;
    }
    template<typename F>
    bool sym(F& f, const char* name) const {
#ifdef _WIN32
        f = reinterpret_cast<F>(reinterpret_cast<void*>(GetProcAddress((HMODULE)h_, name)));
#else
        f = reinterpret_cast<F>(dlsym(h_, name));
#endif
        return f != nullptr;
    }
private:
    void* h_ = nullptr;
};
