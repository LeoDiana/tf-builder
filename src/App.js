import './App.css';
import {
  Card,
  CardBody, CloseButton,
  Input,
  Select
} from '@chakra-ui/react';
import { Fragment, useState } from 'react';
import { v1 as uuid } from 'uuid';
import { DragDropContext, Draggable, Droppable } from 'react-beautiful-dnd';
import * as tf from '@tensorflow/tfjs';
import toast, { Toaster } from 'react-hot-toast';
import { ACTIVATIONS, LAYERS, LAYERS_PARAMS, PADDING, TF_LAYERS, TSP_LAYERS } from './constants';
import * as TSP from 'tensorspace';

const reorder = (list, startIndex, endIndex) => {
  const result = Array.from(list);
  const [removed] = result.splice(startIndex, 1);
  result.splice(endIndex, 0, removed);

  return result;
};

const is2d = (str) => {
  return str.toLowerCase().includes('2d');
};

const parseParams = (params, layerName) => {
  const newParams = { ...params };

  Object.keys(newParams).forEach(param => {
    switch (LAYERS_PARAMS[layerName][param]) {
      case 'number':
        newParams[param] = Number(newParams[param]);
        break;
      case 'number[]':
        if (typeof newParams[param] === 'string') newParams[param] = newParams[param].split(',').map(n => Number(n));
        break;
      case 'number|[number, number]':
        if (typeof newParams[param] === 'string') newParams[param] = newParams[param].split(',').map(n => Number(n));
        break;
      default: throw new Error('Not implemented!')
    }
  });

  return newParams;
};

function App() {
  const [layers, setLayers] = useState([]);
  const [width, setWidth] = useState(0);
  const [channels, setChannels] = useState(0);
  const [output, setOutput] = useState();

  const makeTFModel = () => {
    const model = tf.sequential();
    layers.forEach((layer, index) => {
      const layerParams = { ...parseParams(layer.params, layer.name) };
      if (index === 0) {
        layerParams.inputShape = is2d(layer.name)
          ? [Number(width), Number(width), Number(channels)]
          : [Number(width)];
      }
      model.add(tf.layers[TF_LAYERS[layer.name]](layerParams));
    });
    return model;
  }

  const makeTSPModel = () => {
    const canvas = document.getElementById('tsp-canvas');
    if(canvas && canvas.firstChild) {
      canvas.removeChild(canvas.firstChild);
    }

    const modelTSP = new TSP.models.Sequential( canvas );

    if(!is2d(layers[0].name)) {
      modelTSP.add(new TSP.layers.Input1d({shape: [Number(width)]}));
    } else {
      if (Number(channels) === 2) {
        modelTSP.add(new TSP.layers.GreyscaleInput({shape: [Number(width), Number(width)]}));
      } else { // Number(channels) === 3
        modelTSP.add(new TSP.layers.RGBInput({shape: [Number(width), Number(width), 3]}));
      }
    }

    layers.forEach((layer) => {
      modelTSP.add(new TSP.layers[TSP_LAYERS[layer.name]]({
        name: layer.name,
        ...parseParams(layer.params, layer.name)
      }));
    });

    return modelTSP;
  }

  const handleAddLayer = (layer) => setLayers(layers => [...layers, {
    id: uuid(),
    name: layer,
    params: Object.keys(LAYERS_PARAMS[layer]).reduce((acc, param) => {
      switch (param) {
        case 'activation':
          acc[param] = ACTIVATIONS[0];
          break;
        case 'padding':
          acc[param] = PADDING[0];
          break;
        default:
          acc[param] = undefined;
      }
      return acc;
    }, {})
  }]);

  const handleDeleteLayer = (layerId) => {
    setLayers(layers => layers.filter(l => l.id !== layerId))
  }

  const handleChangeProp = (value, layerId, param) => {
    setLayers(layers => layers.map(
      l => l.id === layerId
        ? { ...l, params: { ...l.params, [param]: value } }
        : l
    ));
  };

  const handleCalculate = () => {
    toast.promise(new Promise((resolve, reject) => {
      try {
        console.log('LAYERS:');
        const model = makeTFModel();
        setOutput('Count Params: ' + model.countParams());
        const modelTSP = makeTSPModel();
        modelTSP.init();
        resolve();
      } catch (err) {
        console.log(err);
        reject(err.message);
      }
    }), {
      loading: 'Loading...',
      success: 'See output',
      error: (err) => err
    });
  };

  const handleDownload = async () => {
    toast.promise(new Promise((resolve, reject) => {
      try {
        const model = makeTFModel();
        model.save('downloads://my-model');
        resolve();
      } catch (err) {
        console.log(err);
        reject(err.message);
      }
    }), {
      loading: 'Saving...',
      success: 'Saved in Downloads',
      error: (err) => err
    });

  };

  const inputField = (type, layer) => {
    switch (type) {
      case 'activation':
        return (
          <Select
            onChange={(e) => handleChangeProp(e.target.value, layer.id, type)}
            defaultValue={ACTIVATIONS[0]}
            placeholder="Select activation">
            {ACTIVATIONS.map((activation) =>
              <option key={activation} value={activation}>{activation}</option>
            )}
          </Select>);
      case 'padding':
        return (
          <Select
            onChange={(e) => handleChangeProp(e.target.value, layer.id, type)}
            defaultValue={PADDING[0]}
            placeholder="Select padding">
            {PADDING.map((padding) =>
              <option key={padding} value={padding}>{padding}</option>
            )}
          </Select>
        );
      default:
        return (<Input
          onChange={(e) => handleChangeProp(e.target.value, layer.id, type)}
          placeholder={type} />);
    }
  };

  const onDragEnd = (result) => {
    // dropped outside the list
    if (!result.destination) {
      return;
    }

    const items = reorder(
      layers,
      result.source.index,
      result.destination.index
    );

    setLayers(items);
  };


  return (
    <div className="app-container">
      <Toaster />
      <h1 className="title">TenzorFlow Calculator</h1>
      <div className="buttons-container">
        <div>Add layer:</div>
        {LAYERS.map(layer =>
          <button className="add-button"
                  key={layer}
                  onClick={() => handleAddLayer(layer)}>
            {layer}
          </button>)}
      </div>
      <div className="input-shape">
        <p className="input-label">Input shape:</p>
        <Input
          onChange={(e) => {
            setWidth(e.target.value);
          }}
          placeholder="Input shape" />
        <p className="input-label">Number of channels:</p>
        <Input
          isDisabled={layers.length ? !is2d(layers[0].name) : true}
          onChange={(e) => {
            setChannels(e.target.value);
          }}
          placeholder="Number of channels" />
        <button
          className="calc-button"
          onClick={handleCalculate}
        >
          Calculate
        </button>
        <button
          className="download-button"
          onClick={handleDownload}
        >
          Download
        </button>
      </div>

      <DragDropContext onDragEnd={onDragEnd}>
        <Droppable droppableId="droppable" direction="horizontal">
          {(provided) => (
            <div
              ref={provided.innerRef}
              className="layers-container"
              {...provided.droppableProps}
            >
              {layers.map((layer, index) => (
                <Draggable key={layer.id} draggableId={layer.id} index={index}>
                  {(provided) => (
                    <Card key={layer.id} width={'300px'} className="layer-card"
                          ref={provided.innerRef}
                          {...provided.draggableProps}
                          {...provided.dragHandleProps}
                    >
                      <CloseButton className='card-delete' onClick={() => handleDeleteLayer(layer.id)}/>
                      <h3 className="card-header">{layer.name}</h3>
                      <CardBody>
                        {Object.keys(layer).length && Object.keys(layer.params).map(param =>
                          (
                            <Fragment key={param}>
                              <p className="input-label">{param}:</p>
                              {inputField(param, layer)}
                            </Fragment>
                          )
                        )}
                      </CardBody>
                    </Card>
                  )}
                </Draggable>
              ))}
              {provided.placeholder}
            </div>
          )}
        </Droppable>
      </DragDropContext>

      <div id="tsp-container">
        <h3>Model Visualisation: </h3>
        <div id="tsp-canvas"></div>
      </div>

      {output && <div className="output-container">
        Output:
        <br />
        {output}
      </div>}
    </div>
  );
}

export default App;
