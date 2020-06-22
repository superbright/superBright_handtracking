/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as handpose from '@tensorflow-models/handpose';
import { TFHUB_SEARCH_PARAM } from '@tensorflow/tfjs-converter/dist/executor/graph_model';
import { math } from '@tensorflow/tfjs-core';
import { HandDetector } from '@tensorflow-models/handpose/dist/hand';

function isMobile() {
  const isAndroid = /Android/i.test(navigator.userAgent);
  const isiOS = /iPhone|iPad|iPod/i.test(navigator.userAgent);
  return isAndroid || isiOS;
}

let model, ctx, videoWidth, videoHeight, scatterGLHasInitialized = false, video, canvas, scatterGL,
  fingerLookupIndices = {
    thumb: [0, 1, 2, 3, 4],
    indexFinger: [0, 5, 6, 7, 8],
    middleFinger: [0, 9, 10, 11, 12],
    ringFinger: [0, 13, 14, 15, 16],
    pinky: [0, 17, 18, 19, 20]
  }; // for rendering each finger as a polyline


// These anchor points allow the hand pointcloud to resize according to its
// position in the input.
// const ANCHOR_POINTS = [[0, 0, 0], [0, -VIDEO_HEIGHT, 0],
// [-VIDEO_WIDTH, 0, 0], [-VIDEO_WIDTH, -VIDEO_HEIGHT, 0]];

const VIDEO_WIDTH = 640;
const VIDEO_HEIGHT = 500;
const mobile = isMobile();
// Don't render the point cloud on mobile in order to maximize performance and
// to avoid crowding limited screen space.
const renderPointcloud = mobile === false;

const state = {};
const stats = new Stats();

var container;
var camera, scene, renderer;
var controls;
var nodeGroup = new THREE.Group(), handLineGroup = new THREE.Group();

if (renderPointcloud) {
  state.renderPointcloud = true;
}

function setupDatGui() {
  const gui = new dat.GUI();

  if (renderPointcloud) {
    gui.add(state, 'renderPointcloud').onChange(render => {
      document.querySelector('#scatter-gl-container').style.display =
        render ? 'inline-block' : 'none';
    });
  }
}

function drawPoint(ctx, y, x, r) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, 2 * Math.PI);
  ctx.fill();
}

function drawKeypoints(ctx, keypoints) {
  const keypointsArray = keypoints;

  for (let i = 0; i < keypointsArray.length; i++) {
    const y = keypointsArray[i][0];
    const x = keypointsArray[i][1];
    drawPoint(ctx, x - 2, y - 2, 3);
  }

  const fingers = Object.keys(fingerLookupIndices);
  for (let i = 0; i < fingers.length; i++) {
    const finger = fingers[i];
    const points = fingerLookupIndices[finger].map(idx => keypoints[idx]);
    drawPath(ctx, points, false);
  }
}

function drawPath(ctx, points, closePath) {
  const region = new Path2D();
  region.moveTo(points[0][0], points[0][1]);
  for (let i = 1; i < points.length; i++) {
    const point = points[i];
    region.lineTo(point[0], point[1]);
  }

  if (closePath) {
    region.closePath();
  }
  ctx.stroke(region);
}

async function setupCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error(
      'Browser API navigator.mediaDevices.getUserMedia not available');
  }

  video = document.getElementById('video');
  const stream = await navigator.mediaDevices.getUserMedia({
    'audio': false,
    'video': {
      facingMode: 'user',
      // Only setting the video to a specified size in order to accommodate a
      // point cloud, so on mobile devices accept the default size.
      width: mobile ? undefined : VIDEO_WIDTH,
      height: mobile ? undefined : VIDEO_HEIGHT
    },
  });
  video.srcObject = stream;

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}

async function loadVideo() {
  video = await setupCamera();
  video.play();
  return video;
}

async function frameLandmarks() {
  stats.begin();
  ctx.drawImage(video, 0, 0, videoWidth, videoHeight, 0, 0, canvas.width, canvas.height);
  const predictions = await model.estimateHands(video);
  if (predictions.length > 0) {
    const result = predictions[0].landmarks;
    drawKeypoints(ctx, result, predictions[0].annotations);

    if (renderPointcloud === true) {
      const pointsData = result.map(point => {
        return [-point[0], -point[1], -point[2]];
      });

      removeMesh(nodeGroup);
      removeMesh(handLineGroup);

      for (let i = 0; i < 21; i++) {
        drawFingerNode(pointsData[i][0] + VIDEO_WIDTH, pointsData[i][1] + VIDEO_HEIGHT, pointsData[i][2]);
      }

      initHandPose();
      scene.add(nodeGroup);
      scene.add(handLineGroup)
    }
  }

  stats.end();
  requestAnimationFrame(frameLandmarks);
};

const main = async () => {
  model = await handpose.load();

  try {
    video = await loadVideo();
  } catch (e) {
    let info = document.getElementById('info');
    info.textContent = e.message;
    info.style.display = 'block';
    throw e;
  }

  landmarksRealTime(video);
}

const landmarksRealTime = async (video) => {
  setupDatGui();

  stats.showPanel(0);
  document.body.appendChild(stats.dom);

  videoWidth = video.videoWidth;
  videoHeight = video.videoHeight;

  canvas = document.getElementById('output');

  canvas.width = videoWidth;
  canvas.height = videoHeight;

  ctx = canvas.getContext('2d');

  video.width = videoWidth;
  video.height = videoHeight;

  ctx.clearRect(0, 0, videoWidth, videoHeight);
  ctx.strokeStyle = "red";
  ctx.fillStyle = "red";

  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);

  frameLandmarks();

  initFingers();
  animate();

};

navigator.getUserMedia = navigator.getUserMedia ||
  navigator.webkitGetUserMedia || navigator.mozGetUserMedia;

main();

function initFingers() {
  container = document.querySelector('#finger-gl-container');
  container.style = `width: ${videoWidth}px; height: ${videoHeight}px`;

  const fov = 50;
  camera = new THREE.PerspectiveCamera(fov, videoWidth / videoHeight, 1, 3000);
  camera.position.set(videoWidth / 2, videoHeight / 2, videoHeight);
  camera.lookAt(videoWidth / 2, videoHeight / 2, 0);
  camera.updateMatrix();

  scene = new THREE.Scene();

  renderer = new THREE.WebGLRenderer({ canvas: container, antialias: true, alpha: true });
  renderer.setClearColor(0x000000, 0);
  renderer.setSize(videoWidth, videoHeight);
  renderer.gammaOutput = true;

  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.target.set(videoWidth / 2, videoHeight / 2, 0);

  const color = 0xFFFFFF;
  const intensity = 1;

  scene.add(new THREE.AmbientLight(0xf0f0f0));
  const light = new THREE.DirectionalLight(color, intensity);
  light.position.set(1, -2, -4);
  scene.add(light);
  console.log(fingerLookupIndices.thumb[1])
}

function drawFingerNode(x, y, z) {
  const spheregeometry = new THREE.SphereBufferGeometry(4, 120, 80);
  const material = new THREE.MeshPhysicalMaterial({
    color: 0x5a5aff,
    metalness: 0.1,
    roughness: 0.2,
    side: THREE.DoubleSide,
  });

  const mesh = new THREE.Mesh(spheregeometry, material);
  mesh.position.set(x, y, z)
  nodeGroup.add(mesh);
}

function initHandPose() {
  var thumbPoints = [], indexFingerPoints = [], middleFingerPoints = [], ringFingerPoints = [], pinkyPoints = [];
  var thumbLineGeo, indexFingerLineGeo, middleFingerLineGeo, ringFingerLineGeo, pinkyPointsLineGeo;
  var thumbLine, indexFingerLine, middleFingerLine, ringFingeLine, pinkyPointsLine;

  drawFingerMesh(thumbPoints, thumbLineGeo, thumbLine, fingerLookupIndices.thumb[1], fingerLookupIndices.thumb[4]);
  drawFingerMesh(indexFingerPoints, indexFingerLineGeo, indexFingerLine, fingerLookupIndices.indexFinger[1], fingerLookupIndices.indexFinger[4]);
  drawFingerMesh(middleFingerPoints, middleFingerLineGeo, middleFingerLine, fingerLookupIndices.middleFinger[1], fingerLookupIndices.middleFinger[4]);
  drawFingerMesh(ringFingerPoints, ringFingerLineGeo, ringFingeLine, fingerLookupIndices.ringFinger[1], fingerLookupIndices.ringFinger[4]);
  drawFingerMesh(pinkyPoints, pinkyPointsLineGeo, pinkyPointsLine, fingerLookupIndices.pinky[1], fingerLookupIndices.pinky[4]);
}

function drawFingerMesh(points, geometry, line, idx1, idx2) {
  var lineMat = new THREE.LineBasicMaterial({ color: 0xff0000 });
  points.push(nodeGroup.children[0].position);
  for (let i = idx1; i <= idx2; i++) {
    points.push(nodeGroup.children[i].position);
  }
  geometry = new THREE.Geometry().setFromPoints(points);
  line = new THREE.Line(geometry, lineMat);
  handLineGroup.add(line);
}

function removeMesh(group) {
  for (let i = group.children.length - 1; i >= 0; i--) {
    group.remove(group.children[i]);
  }
}

function animate() {
  requestAnimationFrame(animate);
  renderer.render(scene, camera);
  stats.update();
  controls.update();
}
