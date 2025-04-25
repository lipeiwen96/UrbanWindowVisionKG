from shapely.geometry import Polygon, GeometryCollection, MultiPolygon, LineString, Point, box, MultiLineString
from shapely import wkt, affinity
from shapely.geometry import Polygon
from art_public_modules.art_data_exchange.data_exchange import ARTShapelyDataExchanger
from art_public_modules.art_data_structure.shapely.vector_2d import Vector2D
from map_system import map_site_creator, tile_system_generator
from map_system.map_site_creator import MapSite
import shapely
from case_constant import Case
import typing
from shapely.ops import unary_union, cascaded_union

# site_buffer_get_road_areas
# overlappeed_center_lines
# compare_center_lines
# from_outline_vec_to_road=>vector

def from_outline_vec_to_road(site: MapSite, simp_factor: int,seg_num:int):
    """
    step1.将site的outline均分节点，发射向量
    step2.得到最近邻的道路点，得到向量中心点
    """
    pass
    raylist, ray_cut_list, lnlist, centerlist = [], [], [], []
    roads = get_road_areas(site, 1)
    roadpoly = unary_union(roads)
    # area = simplify_boundary(site, simp_factor)
    area = site.site_boundary
    # area = site.site_boundary.difference(roadpoly)
    print(area)

    outlines_pts = area.exterior.coords
    outlines = [LineString([outlines_pts[i], outlines_pts[i - 1]]) for i in range(1, len(outlines_pts))]

    for i in range(len(outlines)):
        rlist=line_seg_vec(site, outlines[i], seg_num, 20)
        raylist.extend(rlist)

    for i in range(len(raylist)):
        pt=intersects_rays_road(roadpoly,raylist[i])

        if isinstance(pt,Point)==True:
            ray_x = Point(raylist[i].coords[0]).x
            ray_y = Point(raylist[i].coords[0]).y
            c_x=(pt.x+ray_x)/2
            c_y=(pt.y+ray_y)/2

            pt1=Point(pt)
            pt2=Point([ray_x,ray_y])

            ray_cut=LineString([pt1,pt2])
            ray_cut_list.append(ray_cut)

            center = Point([c_x, c_y])
            centerlist.append(center)

            ln=LineString([Point(raylist[i].coords[0]),pt])
            lnlist.append(ln)
    return lnlist

def get_road_areas(site:MapSite,beffer_dist:int):
    """
    通过buffer_dist得到场地临近的道路polygon_list
    """
    pass
    ped_roads=[]
    tra_roads=[]
    site_polygon=site.site_boundary
    site_detect_polygon=site_polygon.buffer(beffer_dist,cap_style=1, join_style=1, mitre_limit=5.0)
    roads=[]
    for i in range(len(site.road_polygon_list)):
        if site_detect_polygon.intersects(site.road_polygon_list[i].geometry)==True:
            roads.append(site.road_polygon_list[i].geometry)

    # resultlist=[]
    # simplify_polylist(roads,resultlist)

    intersect_id=[]
    for i in range(len(roads)):
        for j in range(len(roads)):
            if i!=j:
                pass
                if roads[i].intersects(roads[j])==True:
                    intersect_id.append(j)
                    event=roads[i].intersection(roads[j])
                    if event.length<10:
                        a=unary_union([roads[i],roads[j]])
                        # print(a)


    for i in range(len(roads)):
        if roads[i].intersects(site_polygon.exterior)==True:
            event=roads[i].intersection(site_detect_polygon.exterior)
            ln = event
            # if event.type!=LineString:
            #     # lnlist=[event.exploded]
            #     # minx,miny,maxx,maxy=get_2_end(lnlist)
            #     minx, miny, maxx, maxy = get_2_end(event)
            #     pt1=Point([minx,miny])
            #     pt2=Point([maxy,maxy])
            #     ln=LineString([pt1,pt2])
            # print(ln)
            dist=ln.length
            if dist/site_detect_polygon.exterior.length>0.1:ped_roads.append(roads[i])
            elif dist/site_detect_polygon.exterior.length<0.01:ped_roads.append(roads[i])
            else:tra_roads.append(roads[i])
    # # for each in ped_roads:print(each)
    # for each in tra_roads:print(each)
    return roads

def intersects_rays_road(poly:Polygon,ray:LineString):
    pass
    event=ray.intersection(poly.exterior)
    if isinstance(event, Point) == True:
        return event

def get_2_end(mln:MultiLineString):
    ptlist=[]
    minx,miny,maxx,maxy=1000,1000,0.001,0.001
    for line in mln:
        ptlist.extend(line.coords)
    for i in range(len(ptlist)):
        if Point(ptlist[i]).x < minx: minx = Point(ptlist[i]).x
        if Point(ptlist[i]).y < miny: miny = Point(ptlist[i]).y
        if Point(ptlist[i]).x > maxx: maxx = Point(ptlist[i]).x
        if Point(ptlist[i]).y > maxy: maxy = Point(ptlist[i]).y
    return minx,miny,maxx,maxy

def overlappeed_center_lines(site:MapSite,selected_roads):
    """
    通过polygon_list得到与其overlap的原始center_lines
    """
    pass
    overlappeed=[]
    for j in range(len(selected_roads)):
        for i in range(len(site.road_center_line_list)):
            if site.road_center_line_list[i].geometry.intersects(selected_roads[j]):
                overlappeed.append(site.road_center_line_list[i].geometry)
    unique_list=[]
    [unique_list.append(item) for item in overlappeed if item not in unique_list]
    return unique_list

def line_seg_vec(site:MapSite,ln:LineString,num:int,l:int):
    pass
    #segmented points
    area=site.site_boundary
    ptlist, endlist,raylist=[],[],[]
    for i in range(num + 1):
        fraction = i / num
        point = Point(ln.interpolate(fraction, normalized=True))
        ptlist.append(point)
    #angle
    ln_rotate=LineString(affinity.rotate(ln,90))
    #ray
    ray_x=(Point(ln_rotate.coords[1]).x-Point(ln_rotate.coords[0]).x)/ln_rotate.length
    ray_y=(Point(ln_rotate.coords[1]).y-Point(ln_rotate.coords[0]).y)/ln_rotate.length
    for i in range(len(ptlist)):
        end=Point([(ptlist[i].x+l*ray_x),(ptlist[i].y+l*ray_y)])
        if end.within(area)==True:
            end=Point([(ptlist[i].x-l*ray_x),(ptlist[i].y-l*ray_y)])
        endlist.append(end)
    raylist=[LineString([ptlist[i], endlist[i]]) for i in range(len(ptlist))]

    return raylist



