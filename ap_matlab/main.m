for i = -167.9735:0.1:-0.0028
    preference = ones(1000,1) * i
    [idx,netsim,dpsim,expref]=apcluster(similarity1000list, preference)
    idx
    unique(idx)
    printf('%d\n',length(unique(idx)))
end
