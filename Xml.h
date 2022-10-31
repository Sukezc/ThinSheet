#pragma once

#include <string>
#include <map>
#include <list>


namespace xml {

class Value
{
public:
    Value();
    Value(bool value);
    Value(int value);
    Value(double value);
    Value(const char * value);
    Value(const std::string & value);
    ~Value();

    Value & operator = (bool value);
    Value & operator = (int value);
    Value & operator = (double value);
    Value & operator = (const char * value);
    Value & operator = (const std::string & value);
    Value & operator = (const Value & value);

    bool operator == (const Value & other);
    bool operator != (const Value & other);
    
    operator bool();
    operator int();
    operator double();
    operator std::string();

private:
    std::string m_value;
};

class Xml
{
public:
    Xml();
    Xml(const char * name);
    Xml(const std::string & name);
    Xml(const Xml & other);
    Xml(const std::string& name, const std::string& text);

    std::string name() const;
    void name(const std::string & name);

    std::string text() const;
    void text(const std::string & text);

    Value & attr(const std::string & key);
    void attr(const std::string & key, const Value & value);

    Xml & operator = (const Xml & other);

    void append(const Xml & child);
    Xml & operator [] (int index);
    Xml & operator [] (const char * name);
    Xml & operator [] (const std::string & name);

    void remove(int index);
    void remove(const char * name);
    void remove(const std::string & name);

    bool check(const char* name);
    bool check(const std::string& name);

    int size() const;

    void clear();

    std::string str(int depth) const;

    typedef std::list<Xml>::iterator iterator;
    iterator begin()
    {
        return m_child->begin();
    }
    iterator end()
    {
        return m_child->end();
    }
    iterator erase(iterator it)
    {
        it->clear();
        return m_child->erase(it);
    }

    bool load(const std::string & filename);
    bool save(const std::string & filename);
    bool parse(const std::string & str);

private:
    std::string * m_name;
    std::string * m_text;
    std::map<std::string, Value> * m_attrs;
    std::list<Xml> * m_child;
};
}
