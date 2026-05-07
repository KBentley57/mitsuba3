#include <mitsuba/core/fresolver.h>
#include <sstream>
#include <algorithm>
#include <cstdlib>
#include <iostream>

#define XSTR(x) #x
#define STR(x) XSTR(x)

NAMESPACE_BEGIN(mitsuba)

FileResolver::FileResolver()
    : Object() {
    m_paths.push_back(fs::current_path());

    #ifdef SUPPORTPACKAGES_MITSUBA_PLUGIN_PATH
        m_paths.push_back(STR(SUPPORTPACKAGES_MITSUBA_PLUGIN_PATH));
    #endif

    if (const char* env_p = std::getenv("MI_PLUGINS_PATH")) {
        if (fs::exists(env_p)) {
            m_paths.push_back(env_p);
        } else {
            std::cerr << "Path does not exist. Check path in env(MI_PLUGINS_PATH)\n";
        }
    }
}

FileResolver::FileResolver(const FileResolver &fr)
  : Object(), m_paths(fr.m_paths) { }

void FileResolver::erase(const fs::path &p) {
    m_paths.erase(std::remove(m_paths.begin(), m_paths.end(), p), m_paths.end());
}

bool FileResolver::contains(const fs::path &p) const {
    return std::find(m_paths.begin(), m_paths.end(), p) != m_paths.end();
}

fs::path FileResolver::resolve(const fs::path &path) const {
    if (!path.is_absolute()) {
        for (auto const &base : m_paths) {
            fs::path combined = base / path;
            if (fs::exists(combined))
                return combined;
        }
    }
    return path;
}

std::string FileResolver::to_string() const {
    std::ostringstream oss;
    oss << "FileResolver[" << std::endl;
    for (size_t i = 0; i < m_paths.size(); ++i) {
        oss << "  \"" << m_paths[i] << "\"";
        if (i + 1 < m_paths.size())
            oss << ",";
        oss << std::endl;
    }
    oss << "]";
    return oss.str();
}

static ref<FileResolver> __static_file_resolver;

void set_file_resolver(FileResolver *file_resolver) { __static_file_resolver = file_resolver; }
FileResolver *file_resolver() { return __static_file_resolver.get(); }

NAMESPACE_END(mitsuba)
