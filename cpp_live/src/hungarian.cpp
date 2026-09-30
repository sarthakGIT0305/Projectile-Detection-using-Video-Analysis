#include "projectile/hungarian.hpp"
#include <algorithm>
#include <limits>
namespace projectile {
std::vector<std::pair<int,int>> hungarian(const std::vector<std::vector<double>>& a){if(a.empty()||a[0].empty())return{};int n=int(a.size()),m=int(a[0].size());if(n>m){std::vector<std::vector<double>> t(m,std::vector<double>(n));for(int i=0;i<n;++i)for(int j=0;j<m;++j)t[j][i]=a[i][j];auto r=hungarian(t);std::vector<std::pair<int,int>> o;for(auto [x,y]:r)o.push_back({y,x});return o;}std::vector<double> u(n+1),v(m+1);std::vector<int> p(m+1),way(m+1);for(int i=1;i<=n;++i){p[0]=i;int j0=0;std::vector<double> minv(m+1,std::numeric_limits<double>::infinity());std::vector<char> used(m+1);do{used[j0]=true;int i0=p[j0],j1=0;double delta=std::numeric_limits<double>::infinity();for(int j=1;j<=m;++j)if(!used[j]){double cur=a[i0-1][j-1]-u[i0]-v[j];if(cur<minv[j])minv[j]=cur,way[j]=j0;if(minv[j]<delta)delta=minv[j],j1=j;}for(int j=0;j<=m;++j)if(used[j])u[p[j]]+=delta,v[j]-=delta;else minv[j]-=delta;j0=j1;}while(p[j0]);do{int j1=way[j0];p[j0]=p[j1];j0=j1;}while(j0); }std::vector<std::pair<int,int>> r;for(int j=1;j<=m;++j)if(p[j])r.push_back({p[j]-1,j-1});return r;}
}
