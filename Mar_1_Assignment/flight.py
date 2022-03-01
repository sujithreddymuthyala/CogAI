g=list(map(int,input().split()))
f=list(map(int,input().split()))
g=sorted(g)
f=sorted(f)


c=0
for i,j in f,g:
    if g[j]<=f[i]:
        c+=1
        