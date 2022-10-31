#pragma once
#include "Xml.h"
#include <string>


namespace xml {

class Parser
{
public:
    Parser();
    bool load_file(const std::string & file);
    bool load_string(const std::string & str);
    Xml parse();

private:
    void skip_white_space();
    bool parse_declaration();
    bool parse_comment();
    Xml parse_element();
    std::string parse_element_name();
    std::string parse_element_text();
    std::string parse_element_attr_key();
    std::string parse_element_attr_val();

private:
    std::string m_str;
    int m_idx;
};

}
