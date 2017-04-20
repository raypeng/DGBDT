/*
   my pretty print << overloads for common stl containers including:
   vector, list, stack, (priority_)queue, (unordered_)set, (unordered_)map
   and newly tuple
*/

#ifndef H_PPRINT
#define H_PPRINT

#include <vector>
#include <list>
#include <stack>
#include <queue>
#include <set>
#include <map>
#include <unordered_set>
#include <unordered_map>

#include <iostream>

#include <tuple>
#include <type_traits>

//#define debug

namespace std {

    template <class T>
    void print(T val, string prefix="") {
#ifdef debug
      cout << prefix << " " << val << endl;
#endif
    }

    template <class U, class V>
    ostream& operator<<(ostream& os, const pair<U,V>& v) {
        os << "(" << v.first << ", " << v.second << ")";
        return os;
    };

    template <class T>
  ostream& operator<<(ostream& os, const vector<T>& v) {
    int i = 0, n = v.size();
    os << "[";
    for (const T& elem : v) {
      os << elem;
      if (i != n - 1) {
        os << ", ";
      }
      i++;
    }
    os << "]";
    return os;
  }

  template <class T>
  ostream& operator<<(ostream& os, const list<T>& v) {
    int i = 0, n = v.size();
    os << "[";
    for (const T& elem : v) {
      os << elem;
      if (i != n - 1) {
        os << ", ";
      }
      i++;
    }
    os << "]";
    return os;
  }

  template <class T>
  ostream& operator<<(ostream& os, const stack<T>& v) {
    stack<T> w = v;
    os << "[";
    T tmp;
    int i = 0, n = v.size();
    while (!w.empty()) {
      tmp = w.top();
      w.pop();
      os << tmp;
      if (i != n - 1) {
        os << ", ";
      }
      i++;
    }
    os << "]";
    return os;
  }

  template <class T>
  ostream& operator<<(ostream& os, const queue<T>& v) {
    queue<T> w = v;
    os << "[";
    T tmp;
    int i = 0, n = v.size();
    while (!w.empty()) {
      tmp = w.front();
      w.pop();
      os << tmp;
      if (i != n - 1) {
        os << ", ";
      }
      i++;
    }
    os << "]";
    return os;
  }

  template < class T,
             class Container = vector<T>,
             class Compare = less<typename Container::value_type> >
  ostream& operator<<(ostream& os, const priority_queue< T, Container, Compare >& v) {
    priority_queue< T, Container, Compare > w = v;
    os << "[";
    T tmp;
    int i = 0, n = v.size();
    while (!w.empty()) {
      tmp = w.top();
      w.pop();
      os << tmp;
      if (i != n - 1) {
        os << ", ";
      }
      i++;
    }
    os << "]";
    return os;
  }

  template <class T>
  ostream& operator<<(ostream& os, const set<T>& v) {
    int i = 0, n = v.size();
    os << "{";
    for (const T& elem : v) {
      os << elem;
      if (i != n - 1) {
        os << ", ";
      }
      i++;
    }
    os << "}";
    return os;
  }

  template <class T>
  ostream& operator<<(ostream& os, const unordered_set<T>& v) {
    int i = 0, n = v.size();
    os << "{";
    for (const T& elem : v) {
      os << elem;
      if (i != n - 1) {
        os << ", ";
      }
      i++;
    }
    os << "}";
    return os;
  }

  template <class T, class W>
  ostream& operator<<(ostream& os, const map<T, W>& v) {
    int i = 0, n = v.size();
    os << "{";
    for (auto& p : v) {
      const T& key = p.first;
      const W& val = p.second;
      os << key << ": " << val;
      if (i != n - 1) {
        os << ", ";
      }
      i++;
    }
    os << "}";
    return os;
  }

  template <class T, class W>
  ostream& operator<<(ostream& os, const unordered_map<T, W>& v) {
    int i = 0, n = v.size();
    os << "{";
    for (auto& p : v) {
      const T& key = p.first;
      const W& val = p.second;
      os << key << ": " << val;
      if (i != n - 1) {
        os << ", ";
      }
      i++;
    }
    os << "}";
    return os;
  }

  template <size_t n, typename... T>
  typename enable_if<(n >= sizeof...(T))>::type
  print_tuple(ostream&, const tuple<T...>&) {
  }

  template <size_t n, typename... T>
  typename enable_if<(n < sizeof...(T))>::type
  print_tuple(ostream& os, const tuple<T...>& tup) {
    if (n != 0) {
      os << ", ";
    }
    os << get<n>(tup);
    print_tuple<n+1>(os, tup);
  }

    template <typename... T>
    ostream& operator<<(ostream& os, const tuple<T...>& tup) {
      os << "<";
      print_tuple<0>(os, tup);
      return os << ">";
    }

}


#endif // H_PPRINT
