from qgis.core import *

# read the UTM zone layer
utm_zone = r'/Users/pratyusht/Downloads/World_UTM_Grid/World_UTM_Grid.shp'
utm_layer = QgsVectorLayer(utm_zone, 'Layer Name', 'ogr')

# loop through each utm rectangle and fragment it into smaller tiles
for feat in utm_layer.getFeatures():
    
    bbox = feat.geometry().boundingBox().toString()
    bbox = bbox.split(' : ')
    bbox = [box.split(',') for box in bbox]
    uid = '_{}{}'.format(feat.attributes()[1], feat.attributes()[2])

    outfile = utm_zone.replace('.shp', f'{uid}.shp')
    params = {'TYPE':2,
              'EXTENT':'{},{},{},{} [EPSG:4326]'.format(
                    bbox[0][0], bbox[1][0], bbox[0][1], bbox[1][1]
              ),
              'HSPACING':1,'VSPACING':1,'HOVERLAY':0,'VOVERLAY':0,
              'CRS':QgsCoordinateReferenceSystem('EPSG:4326'),'OUTPUT':outfile
              }
    result = processing.run("native:creategrid", params)

utm_layer = None