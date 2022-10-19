pt = Point((1, 1))
line = LineString([(0,0), (2,2)])
result = split(line, pt)
result.wkt