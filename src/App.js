import './App.css';
import {
  Card,
  CardBody,
  Input,
  Select,
} from '@chakra-ui/react';
import { Fragment, useState } from 'react';
import { v1 as uuid } from 'uuid';
import { DragDropContext, Draggable, Droppable } from 'react-beautiful-dnd';
import * as tf from '@tensorflow/tfjs';
import toast, { Toaster } from 'react-hot-toast';
import { ACTIVATIONS, LAYERS, LAYERS_PARAMS, PADDING } from './constans';


const reorder = (list, startIndex, endIndex) => {
  const result = Array.from(list);
  const [removed] = result.splice(startIndex, 1);
  result.splice(endIndex, 0, removed);

  return result;
};

const is2d = (str) => {
  return str.toLowerCase().includes('2d');
}

const parseParams = (params, layerName) => {
  const newParams = {...params};

  Object.keys(newParams).forEach(param => {
    switch (LAYERS_PARAMS[layerName][param]) {
      case 'number': newParams[param] = Number(newParams[param]); break;
      case 'number[]': if(typeof newParams[param] === 'string') newParams[param] = newParams[param].split(',').map(n => Number(n)); break;
      case 'number|[number, number]': if(typeof newParams[param] === 'string') newParams[param] = newParams[param].split(',').map(n => Number(n));
    }
  })

  return newParams;
}

function App() {
  const [layers, setLayers] = useState([]);
  const [width, setWidth] = useState(0);
  const [channels, setChannels] = useState(0);
  const [output, setOutput] = useState();
  // const [error, setError] = useState();
  // const [isLoading, setIsLoading] = useState(false);

  // console.log(layers);

  const handleChangeProp = (value, layerId, param) => {
    setLayers(layers => layers.map(
      l => l.id === layerId
        ? {...l, params: {...l.params, [param]: value}}
        : l
    ))
  }

  const handleCalculate = () => {
    toast.promise(new Promise((resolve, reject) => {
      try {
        console.log('LAYERS:');

        const model = tf.sequential();
        layers.forEach((layer, index) => {
          const layerParams = { ...parseParams(layer.params, layer.name) };
          if (index === 0) {
            layerParams.inputShape = is2d(layer.name)
              ? [Number(width), Number(width), Number(channels)]
              : [Number(width)];
          }
          console.log(index, layerParams);
        })
        setOutput('Count Params: ' + model.countParams());
        resolve();
      } catch (err) {
        reject(err.message);
      }
    }), {
      loading: 'Loading',
      success: 'See output',
      error: (err) => err,
    });
  }

  const inputField = (type, layer, param) => {
    switch (type) {
      case 'activation':
        return (
          <Select
            onChange={(e) => handleChangeProp(e.target.value, layer.id, param)}
            placeholder="Select activation">
            {ACTIVATIONS.map((activation, index) =>
              <option selected={index === 0} key={activation} value={activation}>{activation}</option>
            )}
          </Select>);
      case 'padding':
        return (
          <Select
            onChange={(e) => handleChangeProp(e.target.value, layer.id, param)}
            placeholder="Select padding">
            {PADDING.map((padding, index) =>
              <option selected={index === 0} key={padding} value={padding}>{padding}</option>
            )}
          </Select>
        );
      default:
        return (<Input
          onChange={(e) => handleChangeProp(e.target.value, layer.id, param)}
          placeholder={type} />);
    }
  }

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
  }

  const handleAddLayer = (layer) => setLayers(layers => [...layers, {
    id: uuid(),
    name: layer,
    params: Object.keys(LAYERS_PARAMS[layer]).reduce((acc, param) => {
      switch (param) {
        case 'activation': acc[param] = ACTIVATIONS[0]; break;
        case 'padding': acc[param] = PADDING[0]; break;
        default: acc[param] = undefined;
      }
      return acc;
    }, {})
  }])

  return (
    <div className="app-container">
      <Toaster />
      <h1 className='title'>TenzorFlow Calculator</h1>
      <div className="buttons-container">
        <div>Add layer:</div>
        {LAYERS.map(layer =>
          <button className="add-button"
                  key={layer}
                  onClick={() => handleAddLayer(layer)}>
            {layer}
          </button>)}
      </div>
      <div className='input-shape'>
        <p className='input-label'>Input shape:</p>
        <Input
          onChange={(e) => {
            setWidth(e.target.value);
          }}
          placeholder='Input shape' />
        <p className='input-label'>Number of channels:</p>
        <Input
          isDisabled={layers.length ? !is2d(layers[0].name) : true}
          onChange={(e) => {
            setChannels(e.target.value);
          }}
          placeholder='Number of channels' />
      </div>

        <DragDropContext onDragEnd={onDragEnd}>
          <Droppable droppableId="droppable" direction="horizontal">
            {(provided, snapshot) => (
              <div
                ref={provided.innerRef}
                className='layers-container'
                {...provided.droppableProps}
              >
                {layers.map((layer, index) => (
                  <Draggable key={layer.id} draggableId={layer.id} index={index}>
                    {(provided, snapshot) => (
                      <Card key={layer.id} width={'300px'} className="layer-card"
                            ref={provided.innerRef}
                            {...provided.draggableProps}
                            {...provided.dragHandleProps}
                      >
                        <h3 className='card-header'>{layer.name}</h3>
                        <CardBody>
                          {Object.keys(layer).length && Object.keys(layer.params).map(param =>
                            (
                              <Fragment key={param}>
                                <p className='input-label'>{param}:</p>
                                {inputField(param, layer, param)}
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

      <button
        className='calc-button'
        onClick={handleCalculate}
      >
        Calculate
      </button>

      <div className="output-container">
        Output:
        <br/>
        {output}
      </div>
    </div>
  );
}

export default App;