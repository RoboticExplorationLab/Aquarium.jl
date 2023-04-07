function sort_points(x, y)

    xc = sum(x)/length(x)
    yc = sum(y)/length(y)

    angles = rad2deg.(atan.((y .- yc), (x .- xc)))

    p = sortperm(angles)

    return p

end